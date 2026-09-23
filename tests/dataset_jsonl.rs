use std::fs;
use std::path::Path;

use overbrainer::dataset::{
    Appender, DataFiles, DatasetError, Id, Rejected, Rewrite, Subtopic, read, rewrite,
};

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

/// Names of the files in `dir`, sorted.
fn entries(dir: &Path) -> Result<Vec<String>, std::io::Error> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir)? {
        names.push(entry?.file_name().to_string_lossy().into_owned());
    }
    names.sort();
    Ok(names)
}

#[test]
fn a_staged_rewrite_changes_every_file_on_commit() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let files = DataFiles::new(dir.path());
    rewrite(&files.subtopics, &[subtopic("old")])?;
    let mut change = Rewrite::new();
    change.stage(&files.subtopics, &[subtopic("new")])?;
    change.stage(&files.questions, &Vec::<Subtopic>::new())?;
    // Nothing changes before the commit.
    let before: Vec<Subtopic> = read(&files.subtopics)?;
    assert_eq!(before, [subtopic("old")]);
    change.commit()?;
    let after: Vec<Subtopic> = read(&files.subtopics)?;
    assert_eq!(after, [subtopic("new")]);
    assert_eq!(fs::read(&files.questions)?, b"");
    assert_eq!(
        entries(&dir.path().join("data"))?,
        ["questions.jsonl", "subtopics.jsonl"]
    );
    Ok(())
}

#[test]
fn a_failure_while_staging_changes_nothing_and_leaves_no_temp_file()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let files = DataFiles::new(dir.path());
    rewrite(&files.subtopics, &[subtopic("old")])?;
    let original = fs::read(&files.subtopics)?;
    let blocked = dir.path().join("data/blocked");
    fs::write(&blocked, "a file, not a directory")?;
    let mut change = Rewrite::new();
    change.stage(&files.subtopics, &[subtopic("new")])?;
    assert!(
        change
            .stage(&blocked.join("x.jsonl"), &[subtopic("x")])
            .is_err()
    );
    drop(change);
    assert_eq!(fs::read(&files.subtopics)?, original);
    assert_eq!(
        entries(&dir.path().join("data"))?,
        ["blocked", "subtopics.jsonl"]
    );
    Ok(())
}

#[test]
fn commit_renames_in_stage_order_and_stops_at_the_first_failure()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let files = DataFiles::new(dir.path());
    rewrite(&files.subtopics, &[subtopic("old")])?;
    rewrite(&files.answers, &[subtopic("old")])?;
    // A non-empty directory in place of questions.jsonl: its rename fails.
    fs::create_dir_all(files.questions.join("inside"))?;
    let mut change = Rewrite::new();
    change.stage(&files.subtopics, &[subtopic("new")])?;
    change.stage(&files.questions, &[subtopic("new")])?;
    change.stage(&files.answers, &[subtopic("new")])?;
    let error = change.commit();
    assert!(matches!(error, Err(DatasetError::Io { .. })));
    let subtopics: Vec<Subtopic> = read(&files.subtopics)?;
    let answers: Vec<Subtopic> = read(&files.answers)?;
    assert_eq!(subtopics, [subtopic("new")]);
    assert_eq!(answers, [subtopic("old")]);
    assert_eq!(
        entries(&dir.path().join("data"))?,
        ["answers.jsonl", "questions.jsonl", "subtopics.jsonl"]
    );
    Ok(())
}

#[test]
fn rejected_records_round_trip_through_their_file() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let files = DataFiles::new(dir.path());
    assert_eq!(files.rejected, dir.path().join("data/rejected.jsonl"));
    let subtopic_id = Id::subtopic("ownership", "Borrowing");
    let records = [
        Rejected::Subtopic {
            id: subtopic_id.clone(),
            topic: "ownership".into(),
            name: "Borrowing".into(),
        },
        Rejected::Question {
            id: Id::question(&subtopic_id, "Why?"),
            topic: "ownership".into(),
            subtopic_id: subtopic_id.clone(),
            text: "Why?".into(),
        },
    ];
    let mut appender = Appender::open(&files.rejected)?;
    for record in &records {
        appender.append(record)?;
    }
    let content = fs::read_to_string(&files.rejected)?;
    assert!(content.starts_with(&format!(
        "{{\"kind\":\"subtopic\",\"id\":\"{subtopic_id}\",\"topic\":\"ownership\",\"name\":\"Borrowing\"}}\n"
    )));
    let back: Vec<Rejected> = read(&files.rejected)?;
    assert_eq!(back, records);
    assert_eq!(back[1].topic(), "ownership");
    Ok(())
}

#[test]
fn a_rejected_record_with_an_unknown_field_is_refused() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let files = DataFiles::new(dir.path());
    fs::create_dir_all(dir.path().join("data"))?;
    fs::write(
        &files.rejected,
        "{\"kind\":\"subtopic\",\"id\":\"x\",\"topic\":\"t\",\"name\":\"n\",\"extra\":1}\n",
    )?;
    let read_back: Result<Vec<Rejected>, _> = read(&files.rejected);
    assert!(matches!(
        read_back,
        Err(DatasetError::Parse { line: 1, .. })
    ));
    Ok(())
}
