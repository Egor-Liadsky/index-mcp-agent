//! Чтение `.docx`: абзацы с уровнем заголовка и смещениями в общем тексте.
//!
//! `.docx` — zip с XML внутри. Текст берётся только из `word/document.xml`:
//! сноски (`word/footnotes.xml`) и колонтитулы (`word/header*.xml`,
//! `word/footer*.xml`) лежат в отдельных частях и поэтому пропускаются
//! сами собой, а ссылка на сноску в теле текста не несёт.
//!
//! Уровень заголовка определяется по `styles.xml`, а не по `styleId`: в
//! русском Word id стиля «Заголовок 1» — `1`, в английском — `Heading1`,
//! в документах LibreOffice — третье. Надёжны лишь имя стиля
//! (`heading 1`…`heading 9`, его Word пишет по-английски в любой
//! локализации) и `w:outlineLvl`, в том числе унаследованные через
//! `w:basedOn`.

use anyhow::{Context, Result, bail};
use quick_xml::Reader;
use quick_xml::escape::resolve_predefined_entity;
use quick_xml::events::{BytesStart, Event};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Разделитель уровней в пути раздела: `Раздел > Подраздел`.
pub const SECTION_SEPARATOR: &str = " > ";

/// Абзац документа. Смещения — в символах (`char`) внутри [`Document::text`].
#[derive(Debug, Clone, PartialEq)]
pub struct Paragraph {
    pub text: String,
    /// Уровень заголовка 1…9 или `None` для обычного текста.
    pub heading: Option<u8>,
    pub start: usize,
    pub end: usize,
    /// Путь заголовков, под которым стоит абзац; у заголовка — включая его самого.
    pub section: String,
}

/// Разобранный документ: общий текст (абзацы через `\n`) и абзацы в нём.
#[derive(Debug, Clone)]
pub struct Document {
    pub source: PathBuf,
    pub title: String,
    pub text: String,
    pub paragraphs: Vec<Paragraph>,
    /// Символы текста: стратегиям нужен произвольный доступ по индексу символа.
    chars: Vec<char>,
    /// Байтовое смещение каждого символа (и конца текста) — для срезов `&str`.
    byte_at: Vec<usize>,
}

impl Document {
    /// Собирает документ из абзацев: считает смещения и пути разделов.
    /// Пустые после нормализации абзацы отбрасываются.
    pub fn from_paragraphs(source: PathBuf, title: String, raw: Vec<(String, Option<u8>)>) -> Self {
        let mut text = String::new();
        let mut paragraphs = Vec::new();
        let mut stack: Vec<(u8, String)> = Vec::new();
        let mut pos = 0usize;
        for (raw_text, heading) in raw {
            let para_text = normalize_space(&raw_text);
            if para_text.is_empty() {
                continue;
            }
            if let Some(level) = heading {
                while stack.last().is_some_and(|(l, _)| *l >= level) {
                    stack.pop();
                }
                stack.push((level, para_text.clone()));
            }
            let section = stack
                .iter()
                .map(|(_, t)| t.as_str())
                .collect::<Vec<_>>()
                .join(SECTION_SEPARATOR);
            if !text.is_empty() {
                text.push('\n');
                pos += 1;
            }
            let len = para_text.chars().count();
            text.push_str(&para_text);
            paragraphs.push(Paragraph {
                text: para_text,
                heading,
                start: pos,
                end: pos + len,
                section,
            });
            pos += len;
        }
        let chars: Vec<char> = text.chars().collect();
        let mut byte_at: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        byte_at.push(text.len());
        Self {
            source,
            title,
            text,
            paragraphs,
            chars,
            byte_at,
        }
    }

    pub fn char_len(&self) -> usize {
        self.chars.len()
    }

    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// Текст между символьными смещениями `[start, end)`.
    pub fn slice(&self, start: usize, end: usize) -> &str {
        &self.text[self.byte_at[start]..self.byte_at[end]]
    }

    /// Путь раздела в позиции: раздел последнего абзаца, начавшегося не позже `pos`.
    pub fn section_at(&self, pos: usize) -> &str {
        let idx = self.paragraphs.partition_point(|p| p.start <= pos);
        if idx == 0 {
            ""
        } else {
            &self.paragraphs[idx - 1].section
        }
    }

    /// Имя файла без расширения — часть `chunk_id`.
    pub fn file_stem(&self) -> String {
        file_stem(&self.source)
    }
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Схлопывает пробельные последовательности в один пробел и обрезает края:
/// в Word пробелы, табуляции и разрывы строк внутри абзаца для поиска равноценны.
fn normalize_space(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Читает `.docx` с диска.
pub fn read_docx(path: &Path) -> Result<Document> {
    let file =
        std::fs::File::open(path).with_context(|| format!("не открыть {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("{} — не zip-архив .docx", path.display()))?;
    let document = read_part(&mut archive, "word/document.xml")?
        .with_context(|| format!("{}: нет word/document.xml", path.display()))?;
    let styles = match read_part(&mut archive, "word/styles.xml")? {
        Some(xml) => {
            parse_styles(&xml).with_context(|| format!("{}: styles.xml", path.display()))?
        }
        None => HashMap::new(),
    };
    let title = match read_part(&mut archive, "docProps/core.xml")? {
        Some(xml) => parse_title(&xml)?,
        None => None,
    }
    .unwrap_or_else(|| file_stem(path));
    let raw = parse_body(&document, &styles)
        .with_context(|| format!("{}: document.xml", path.display()))?;
    Ok(Document::from_paragraphs(path.to_path_buf(), title, raw))
}

fn read_part<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Option<String>> {
    let mut entry = match archive.by_name(name) {
        Ok(entry) => entry,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("не прочитать {name}")),
    };
    let mut xml = String::new();
    entry
        .read_to_string(&mut xml)
        .with_context(|| format!("{name} — не UTF-8"))?;
    Ok(Some(xml))
}

fn attr(e: &BytesStart<'_>, name: &str) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| a.key.as_ref() == name)
        .map(|a| a.value.into_owned())
}

/// Уровень по имени стиля: `heading 2` → 2. Русское `заголовок 2` встречается
/// в документах, где имя стиля локализовал не Word, а конвертер.
fn level_from_name(name: &str) -> Option<u8> {
    let lower = name.to_lowercase();
    let rest = lower
        .strip_prefix("heading ")
        .or_else(|| lower.strip_prefix("заголовок "))?;
    rest.trim()
        .parse::<u8>()
        .ok()
        .filter(|l| (1..=9).contains(l))
}

/// `w:outlineLvl` нумерует с нуля, `9` — «основной текст».
fn level_from_outline(val: &str) -> Option<u8> {
    val.parse::<u8>().ok().filter(|v| *v < 9).map(|v| v + 1)
}

#[derive(Default)]
struct StyleDef {
    name_level: Option<u8>,
    outline_level: Option<u8>,
    based_on: Option<String>,
}

/// `styleId` абзацного стиля → уровень заголовка. Стили без уровня в карту
/// не попадают.
fn parse_styles(xml: &str) -> Result<HashMap<String, u8>> {
    let mut reader = Reader::from_str(xml);
    let mut defs: HashMap<String, StyleDef> = HashMap::new();
    let mut current: Option<(String, StyleDef)> = None;
    loop {
        match reader.read_event()? {
            Event::Start(e) | Event::Empty(e) => match e.name().as_ref() {
                "w:style" => {
                    let is_paragraph = attr(&e, "w:type").as_deref() == Some("paragraph");
                    current = match (is_paragraph, attr(&e, "w:styleId")) {
                        (true, Some(id)) => Some((id, StyleDef::default())),
                        _ => None,
                    };
                }
                "w:name" => {
                    if let (Some((_, def)), Some(v)) = (current.as_mut(), attr(&e, "w:val")) {
                        def.name_level = level_from_name(&v);
                    }
                }
                "w:basedOn" => {
                    if let Some((_, def)) = current.as_mut() {
                        def.based_on = attr(&e, "w:val");
                    }
                }
                "w:outlineLvl" => {
                    if let (Some((_, def)), Some(v)) = (current.as_mut(), attr(&e, "w:val")) {
                        def.outline_level = level_from_outline(&v);
                    }
                }
                _ => {}
            },
            Event::End(e) if e.name().as_ref() == "w:style" => {
                if let Some((id, def)) = current.take() {
                    defs.insert(id, def);
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    let mut levels = HashMap::new();
    for id in defs.keys() {
        if let Some(level) = resolve_level(&defs, id) {
            levels.insert(id.clone(), level);
        }
    }
    Ok(levels)
}

/// Уровень стиля с учётом наследования; цепочка `basedOn` ограничена, чтобы
/// испорченный `styles.xml` с циклом не зациклил разбор.
fn resolve_level(defs: &HashMap<String, StyleDef>, id: &str) -> Option<u8> {
    let mut current = id;
    for _ in 0..32 {
        let def = defs.get(current)?;
        if let Some(level) = def.name_level.or(def.outline_level) {
            return Some(level);
        }
        current = def.based_on.as_deref()?;
    }
    None
}

fn parse_title(xml: &str) -> Result<Option<String>> {
    let mut reader = Reader::from_str(xml);
    let mut in_title = false;
    let mut title = String::new();
    loop {
        match reader.read_event()? {
            Event::Start(e) if e.local_name().as_ref() == "title" => in_title = true,
            Event::End(e) if e.local_name().as_ref() == "title" => in_title = false,
            Event::Text(t) if in_title => title.push_str(&t.xml10_content()),
            Event::GeneralRef(r) if in_title => push_ref(&mut title, &r),
            Event::Eof => break,
            _ => {}
        }
    }
    let title = normalize_space(&title);
    Ok((!title.is_empty()).then_some(title))
}

fn push_ref(out: &mut String, r: &quick_xml::events::BytesRef<'_>) {
    if let Ok(Some(c)) = r.resolve_char_ref() {
        out.push(c);
    } else if let Some(s) = resolve_predefined_entity(r) {
        out.push_str(s);
    }
}

/// Состояние разбора `document.xml`.
#[derive(Default)]
struct BodyState {
    out: Vec<(String, Option<u8>)>,
    /// Глубина вложенности `w:p`: абзацы внутри надписей (`w:txbxContent`)
    /// лежат внутри внешнего абзаца и дописываются в него.
    p_depth: usize,
    in_ppr: bool,
    in_text: bool,
    paragraph: String,
    style: Option<String>,
    outline: Option<u8>,
    tbl_depth: usize,
    row: Vec<String>,
    cell: String,
    /// Внутри `mc:Fallback` и `w:moveFrom`: запасная копия надписи и
    /// перемещённый (удалённый) текст — иначе текст задвоится.
    skip_depth: usize,
}

impl BodyState {
    fn finish_paragraph(&mut self, styles: &HashMap<String, u8>) {
        let text = std::mem::take(&mut self.paragraph);
        let level = self
            .outline
            .take()
            .or_else(|| self.style.take().and_then(|s| styles.get(&s).copied()));
        self.style = None;
        if self.tbl_depth > 0 {
            // Абзацы ячейки склеиваются в одну строку ячейки.
            if !self.cell.is_empty() {
                self.cell.push(' ');
            }
            self.cell.push_str(&text);
        } else {
            self.out.push((text, level));
        }
    }
}

fn parse_body(xml: &str, styles: &HashMap<String, u8>) -> Result<Vec<(String, Option<u8>)>> {
    let mut reader = Reader::from_str(xml);
    let mut st = BodyState::default();
    loop {
        let event = reader.read_event()?;
        if st.skip_depth > 0 {
            match &event {
                Event::Start(e) if is_skipped(e.name().as_ref()) => st.skip_depth += 1,
                Event::End(e) if is_skipped(e.name().as_ref()) => st.skip_depth -= 1,
                Event::Eof => bail!("документ оборвался внутри mc:Fallback"),
                _ => {}
            }
            continue;
        }
        match event {
            Event::Start(e) => match e.name().as_ref() {
                name if is_skipped(name) => st.skip_depth = 1,
                "w:p" => {
                    st.p_depth += 1;
                    if st.p_depth == 1 {
                        st.paragraph.clear();
                        st.style = None;
                        st.outline = None;
                    }
                }
                "w:pPr" => st.in_ppr = true,
                "w:t" => st.in_text = true,
                "w:tbl" => st.tbl_depth += 1,
                "w:tr" if st.tbl_depth == 1 => st.row.clear(),
                "w:tc" if st.tbl_depth == 1 => st.cell.clear(),
                _ => {}
            },
            Event::Empty(e) => match e.name().as_ref() {
                "w:pStyle" if st.in_ppr && st.p_depth == 1 => st.style = attr(&e, "w:val"),
                "w:outlineLvl" if st.in_ppr && st.p_depth == 1 => {
                    st.outline = attr(&e, "w:val").as_deref().and_then(level_from_outline);
                }
                "w:tab" | "w:br" | "w:cr" if st.p_depth > 0 && !st.in_ppr => st.paragraph.push(' '),
                "w:noBreakHyphen" if st.p_depth > 0 => st.paragraph.push('-'),
                _ => {}
            },
            Event::End(e) => match e.name().as_ref() {
                "w:p" => {
                    st.p_depth = st.p_depth.saturating_sub(1);
                    if st.p_depth == 0 {
                        st.finish_paragraph(styles);
                    }
                }
                "w:pPr" => st.in_ppr = false,
                "w:t" => st.in_text = false,
                "w:tbl" => st.tbl_depth = st.tbl_depth.saturating_sub(1),
                "w:tc" if st.tbl_depth == 1 => {
                    let cell = normalize_space(&std::mem::take(&mut st.cell));
                    st.row.push(cell);
                }
                "w:tr" if st.tbl_depth == 1 => {
                    let row = std::mem::take(&mut st.row);
                    if row.iter().any(|c| !c.is_empty()) {
                        st.out.push((row.join(" | "), None));
                    }
                }
                _ => {}
            },
            Event::Text(t) if st.in_text => st.paragraph.push_str(&t.xml10_content()),
            Event::GeneralRef(r) if st.in_text => push_ref(&mut st.paragraph, &r),
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(st.out)
}

fn is_skipped(name: &str) -> bool {
    matches!(name, "mc:Fallback" | "w:moveFrom")
}

#[cfg(test)]
pub mod testutil {
    //! Сборка `.docx` в памяти: тесты не держат двоичных файлов в репозитории.

    use std::io::Write;
    use std::path::Path;

    const NS: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006""#;

    /// Стили как в русском Word: id `1`, `2`, `3`, имя `heading N`.
    pub fn russian_styles() -> String {
        let mut s = String::new();
        for level in 1..=3 {
            s.push_str(&format!(
                r#"<w:style w:type="paragraph" w:styleId="{level}"><w:name w:val="heading {level}"/><w:basedOn w:val="a"/><w:pPr><w:outlineLvl w:val="{}"/></w:pPr></w:style>"#,
                level - 1
            ));
        }
        s.push_str(r#"<w:style w:type="paragraph" w:default="1" w:styleId="a"><w:name w:val="Normal"/></w:style>"#);
        s
    }

    pub fn heading(level: u8, text: &str) -> String {
        format!(
            r#"<w:p><w:pPr><w:pStyle w:val="{level}"/></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"#
        )
    }

    pub fn para(text: &str) -> String {
        format!(r#"<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>"#)
    }

    pub fn docx_bytes(styles: &str, body: &str, title: Option<&str>) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts = zip::write::SimpleFileOptions::default();
            zip.start_file("word/document.xml", opts).unwrap();
            write!(
                zip,
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {NS}><w:body>{body}<w:sectPr/></w:body></w:document>"#
            )
            .unwrap();
            zip.start_file("word/styles.xml", opts).unwrap();
            write!(
                zip,
                r#"<?xml version="1.0"?><w:styles {NS}>{styles}</w:styles>"#
            )
            .unwrap();
            // Сноски и колонтитулы кладутся в архив, чтобы тест видел, что их текст не попадает.
            zip.start_file("word/footnotes.xml", opts).unwrap();
            write!(zip, r#"<w:footnotes {NS}><w:footnote w:id="1"><w:p><w:r><w:t>СНОСКА</w:t></w:r></w:p></w:footnote></w:footnotes>"#).unwrap();
            zip.start_file("word/header1.xml", opts).unwrap();
            write!(
                zip,
                r#"<w:hdr {NS}><w:p><w:r><w:t>КОЛОНТИТУЛ</w:t></w:r></w:p></w:hdr>"#
            )
            .unwrap();
            if let Some(title) = title {
                zip.start_file("docProps/core.xml", opts).unwrap();
                write!(zip, r#"<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title></cp:coreProperties>"#).unwrap();
            }
            zip.finish().unwrap();
        }
        buf.into_inner()
    }

    pub fn write_docx(path: &Path, styles: &str, body: &str, title: Option<&str>) {
        std::fs::write(path, docx_bytes(styles, body, title)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::*;
    use super::*;

    fn read(body: &str, styles: &str, title: Option<&str>) -> Document {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("конспект.docx");
        write_docx(&path, styles, body, title);
        read_docx(&path).unwrap()
    }

    #[test]
    fn russian_style_ids_resolve_by_name() {
        let body = [
            heading(1, "Сети"),
            para("Вводный абзац."),
            heading(2, "TCP"),
            para("Надёжная доставка."),
            heading(1, "ОС"),
        ]
        .concat();
        let doc = read(&body, &russian_styles(), None);
        let levels: Vec<_> = doc.paragraphs.iter().map(|p| p.heading).collect();
        assert_eq!(levels, [Some(1), None, Some(2), None, Some(1)]);
        let sections: Vec<_> = doc.paragraphs.iter().map(|p| p.section.as_str()).collect();
        assert_eq!(sections, ["Сети", "Сети", "Сети > TCP", "Сети > TCP", "ОС"]);
    }

    #[test]
    fn outline_level_and_based_on_are_inherited() {
        let styles = r#"
            <w:style w:type="paragraph" w:styleId="MyTop"><w:name w:val="Мой заголовок"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>
            <w:style w:type="paragraph" w:styleId="Child"><w:name w:val="Потомок"/><w:basedOn w:val="MyTop"/></w:style>
            <w:style w:type="paragraph" w:styleId="Body"><w:name w:val="Body"/><w:pPr><w:outlineLvl w:val="9"/></w:pPr></w:style>
            <w:style w:type="character" w:styleId="1"><w:name w:val="heading 1"/></w:style>"#;
        let body = [
            r#"<w:p><w:pPr><w:pStyle w:val="Child"/></w:pPr><w:r><w:t>А</w:t></w:r></w:p>"#,
            r#"<w:p><w:pPr><w:pStyle w:val="Body"/></w:pPr><w:r><w:t>Б</w:t></w:r></w:p>"#,
            r#"<w:p><w:pPr><w:outlineLvl w:val="2"/></w:pPr><w:r><w:t>В</w:t></w:r></w:p>"#,
            r#"<w:p><w:pPr><w:pStyle w:val="1"/></w:pPr><w:r><w:t>Г</w:t></w:r></w:p>"#,
        ]
        .concat();
        let doc = read(&body, styles, None);
        let levels: Vec<_> = doc.paragraphs.iter().map(|p| p.heading).collect();
        // Символьный стиль с именем heading 1 абзац заголовком не делает.
        assert_eq!(levels, [Some(1), None, Some(3), None]);
    }

    #[test]
    fn tables_become_pipe_rows_and_side_parts_are_skipped() {
        let body = [
            para("До"),
            r#"<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Протокол</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Порт</w:t></w:r></w:p></w:tc></w:tr>
               <w:tr><w:tc><w:p><w:r><w:t>HTTP</w:t></w:r></w:p><w:p><w:r><w:t>1.1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>80</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"#.to_string(),
            r#"<w:p><w:r><w:t>Текст</w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r><w:r><w:t xml:space="preserve"> и &amp; &#1046;</w:t></w:r></w:p>"#.to_string(),
        ]
        .concat();
        let doc = read(&body, &russian_styles(), None);
        let texts: Vec<_> = doc.paragraphs.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(
            texts,
            ["До", "Протокол | Порт", "HTTP 1.1 | 80", "Текст и & Ж"]
        );
        assert!(!doc.text.contains("СНОСКА"));
        assert!(!doc.text.contains("КОЛОНТИТУЛ"));
    }

    #[test]
    fn fallback_copies_and_moved_text_are_not_duplicated() {
        let body = r#"<w:p><w:r><w:t>Раз</w:t></w:r><mc:AlternateContent><mc:Choice><w:r><w:t xml:space="preserve"> два</w:t></w:r></mc:Choice><mc:Fallback><w:r><w:t>ДУБЛЬ</w:t></w:r></mc:Fallback></mc:AlternateContent><w:moveFrom><w:r><w:t>СТАРОЕ</w:t></w:r></w:moveFrom><w:r><w:tab/><w:t>три</w:t></w:r></w:p>"#;
        let doc = read(body, "", None);
        assert_eq!(doc.text, "Раз два три");
    }

    #[test]
    fn offsets_are_char_based_and_slice_back() {
        let body = [
            heading(1, "Заголовок"),
            para("  Ёлка   и  ель "),
            para("Третий"),
        ]
        .concat();
        let doc = read(&body, &russian_styles(), None);
        assert_eq!(doc.text, "Заголовок\nЁлка и ель\nТретий");
        for p in &doc.paragraphs {
            assert_eq!(doc.slice(p.start, p.end), p.text);
        }
        assert_eq!(doc.paragraphs[1].start, 10);
        assert_eq!(doc.section_at(12), "Заголовок");
    }

    #[test]
    fn title_comes_from_core_or_file_name() {
        assert_eq!(read(&para("x"), "", Some("Сети ЭВМ")).title, "Сети ЭВМ");
        assert_eq!(read(&para("x"), "", Some("  ")).title, "конспект");
        assert_eq!(read(&para("x"), "", None).title, "конспект");
    }

    #[test]
    fn not_a_docx_is_an_error_with_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.docx");
        std::fs::write(&path, b"not zip").unwrap();
        let err = format!("{:#}", read_docx(&path).unwrap_err());
        assert!(err.contains("broken.docx"), "{err}");
    }
}
