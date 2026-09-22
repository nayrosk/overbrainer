use overbrainer::dataset::{Appender, DataFiles, DatasetError, Id, Subtopic, read, rewrite};

fn subtopic(name: &str) -> Subtopic {
    Subtopic {
        id: Id::subtopic("ownership", name),
        topic: "ownership".into(),
        name: name.into(),
    }
}

#[test]
fn missing_file_reads_as_empty() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let items: Vec<Subtopic> = read(&dir.path().join("none.jsonl"))?;
    assert!(items.is_empty());
    Ok(())
}

#[test]
fn appended_lines_are_read_back_in_order() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let files = DataFiles::new(dir.path());
    let mut appender = Appender::open(&files.subtopics)?;
    appender.append(&subtopic("borrowing"))?;
    appender.append(&subtopic("lifetimes"))?;
    drop(appender);
    let mut appender = Appender::open(&files.subtopics)?;
    appender.append(&subtopic("moves"))?;
    let items: Vec<Subtopic> = read(&files.subtopics)?;
    let names: Vec<&str> = items.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["borrowing", "lifetimes", "moves"]);
    let content = std::fs::read_to_string(&files.subtopics)?;
    assert_eq!(content.lines().count(), 3);
    assert!(content.ends_with('\n'));
    Ok(())
}

#[test]
fn incomplete_last_line_is_skipped_then_removed() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("subtopics.jsonl");
    let complete = serde_json::to_string(&subtopic("borrowing"))?;
    std::fs::write(&path, format!("{complete}\n{{\"id\":\"trunc"))?;
    let items: Vec<Subtopic> = read(&path)?;
    assert_eq!(items.len(), 1);

    let mut appender = Appender::open(&path)?;
    appender.append(&subtopic("moves"))?;
    let items: Vec<Subtopic> = read(&path)?;
    let names: Vec<&str> = items.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["borrowing", "moves"]);
    Ok(())
}

#[test]
fn a_corrupt_complete_line_is_an_error_with_its_line_number()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("subtopics.jsonl");
    let complete = serde_json::to_string(&subtopic("borrowing"))?;
    std::fs::write(&path, format!("{complete}\nnot json\n{complete}\n"))?;
    match read::<Subtopic>(&path) {
        Err(DatasetError::Parse { line, .. }) => {
            assert_eq!(line, 2);
            Ok(())
        },
        other => Err(format!("expected Parse, got {other:?}").into()),
    }
}

#[test]
fn rewrite_replaces_content_and_leaves_no_temp_file() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let files = DataFiles::new(dir.path());
    let mut appender = Appender::open(&files.subtopics)?;
    appender.append(&subtopic("borrowing"))?;
    appender.append(&subtopic("lifetimes"))?;
    drop(appender);

    rewrite(&files.subtopics, &[subtopic("moves")])?;
    let items: Vec<Subtopic> = read(&files.subtopics)?;
    assert_eq!(items, vec![subtopic("moves")]);
    let leftovers: Vec<_> = std::fs::read_dir(dir.path().join("data"))?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty());
    Ok(())
}

#[test]
fn a_valid_last_line_without_newline_survives_open_and_append()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("subtopics.jsonl");
    let first = serde_json::to_string(&subtopic("borrowing"))?;
    let last = serde_json::to_string(&subtopic("lifetimes"))?;
    std::fs::write(&path, format!("{first}\n{last}"))?;
    let items: Vec<Subtopic> = read(&path)?;
    assert_eq!(items.len(), 2, "the unterminated but valid line is read");

    let mut appender = Appender::open(&path)?;
    appender.append(&subtopic("moves"))?;
    let items: Vec<Subtopic> = read(&path)?;
    let names: Vec<&str> = items.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["borrowing", "lifetimes", "moves"]);
    let content = std::fs::read_to_string(&path)?;
    assert_eq!(content.lines().count(), 3);
    assert!(content.ends_with('\n'));
    Ok(())
}

#[test]
fn an_unterminated_last_line_that_is_json_but_not_a_record_is_an_error()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("subtopics.jsonl");
    let complete = serde_json::to_string(&subtopic("borrowing"))?;
    std::fs::write(&path, format!("{complete}\n{{\"other\":1}}"))?;
    match read::<Subtopic>(&path) {
        Err(DatasetError::Parse { line, .. }) => {
            assert_eq!(line, 2);
            Ok(())
        },
        other => Err(format!("expected Parse, got {other:?}").into()),
    }
}

#[test]
fn a_tail_cut_inside_a_multibyte_character_is_skipped_then_removed()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("subtopics.jsonl");
    let complete = serde_json::to_string(&subtopic("borrowing"))?;
    let cut = serde_json::to_string(&subtopic("émoi"))?;
    let accent = cut.find('é').ok_or("no accent")?;
    let mut bytes = format!("{complete}\n").into_bytes();
    bytes.extend_from_slice(&cut.as_bytes()[..=accent]);
    std::fs::write(&path, bytes)?;
    let items: Vec<Subtopic> = read(&path)?;
    assert_eq!(items, vec![subtopic("borrowing")]);

    let mut appender = Appender::open(&path)?;
    appender.append(&subtopic("moves"))?;
    let items: Vec<Subtopic> = read(&path)?;
    let names: Vec<&str> = items.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["borrowing", "moves"]);
    Ok(())
}
