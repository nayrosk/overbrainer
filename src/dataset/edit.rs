//! Edits of the dataset files, as the terminal UI makes them: editing a question, an
//! answer or a subtopic name, with the IDs recomputed and the dependent records kept
//! consistent, and deleting with a cascade. Every operation works on a freshly read [`Dataset`] and checks first
//! that what it changes is still what the user saw; [`Dataset::save`] then writes
//! every touched file at once.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    DataFiles, DatasetError, Example, Id, Question, Rejected, Rewrite, Role, Subtopic, read,
};

/// Why an edit was refused. Messages hold dataset text only.
#[derive(Debug, thiserror::Error)]
pub enum EditError {
    /// A data file cannot be read or written.
    #[error(transparent)]
    Dataset(#[from] DatasetError),
    /// What the edit changes is no longer on disk as the user saw it.
    #[error("changed on disk since the edit began; reloaded")]
    Changed,
    /// The edited question's new ID is another question's.
    #[error("another question of this subtopic already has this text")]
    QuestionExists,
    /// The renamed subtopic's new ID is another subtopic's.
    #[error("a subtopic with this name already exists in topic {topic}")]
    SubtopicExists {
        /// The topic of both subtopics.
        topic: String,
    },
    /// A subtopic name spans several lines.
    #[error("a subtopic name must be one line")]
    MultiLine,
    /// Question IDs recomputed by a rename are already used by other records.
    #[error(
        "{count} recomputed question ID(s) are already used in data/questions.jsonl or data/answers.jsonl; nothing changed"
    )]
    Rekey {
        /// How many recomputed IDs are taken.
        count: usize,
    },
    /// The edit empties the reasoning of an answer that has one.
    #[error("emptying the reasoning of an answer is refused: delete the answer instead")]
    ReasoningRemoved,
    /// A deletion would remove other counts than the confirmed ones.
    #[error("changed on disk; reloaded; press d again")]
    CountsChanged,
}

/// What a deletion removes, with what depends on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Deletion {
    /// A subtopic, its questions and their answers; the subtopic is recorded as
    /// rejected.
    Subtopic(Id),
    /// A question and its answer; the question is recorded as rejected.
    Question(Id),
    /// An answer only.
    Answer(Id),
    /// The questions of a topic whose subtopic no longer exists, and their answers.
    MissingSubtopic(String),
}

/// How many questions and answers a [`Deletion`] removes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Questions removed.
    pub questions: usize,
    /// Answers removed.
    pub answers: usize,
}

/// Which dataset files a change rewrites.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Touched(u8);

impl Touched {
    /// No file.
    pub const NONE: Self = Self(0);
    /// `data/subtopics.jsonl`
    pub const SUBTOPICS: Self = Self(1);
    /// `data/questions.jsonl`
    pub const QUESTIONS: Self = Self(2);
    /// `data/answers.jsonl`
    pub const ANSWERS: Self = Self(4);
    /// `data/rejected.jsonl`
    pub const REJECTED: Self = Self(8);
    /// Every file.
    pub const ALL: Self = Self(15);

    /// The files of both.
    #[must_use]
    pub const fn and(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every file of `other` is among these.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// What an applied edit changed, for [`Dataset::save`] and the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    /// The files to rewrite.
    pub touched: Touched,
    /// Records to append to `data/rejected.jsonl`.
    pub append: Vec<Rejected>,
    /// One line saying what was done.
    pub message: String,
}

impl Change {
    fn new(touched: Touched, message: impl Into<String>) -> Self {
        Self {
            touched,
            append: Vec::new(),
            message: message.into(),
        }
    }
}

/// The assistant message of an answer: its reasoning, when it has one, and content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerText {
    /// `reasoning_content`.
    pub reasoning: Option<String>,
    /// `content`.
    pub content: String,
}

impl AnswerText {
    /// The assistant message of `example`, if it has one.
    #[must_use]
    pub fn of(example: &Example) -> Option<Self> {
        let message = example
            .messages
            .iter()
            .find(|message| message.role == Role::Assistant)?;
        Some(Self {
            reasoning: message.reasoning_content.clone(),
            content: message.content.clone(),
        })
    }
}

/// The dataset files an edit reads and rewrites.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Dataset {
    /// `data/subtopics.jsonl`
    pub subtopics: Vec<Subtopic>,
    /// `data/questions.jsonl`
    pub questions: Vec<Question>,
    /// `data/answers.jsonl`
    pub answers: Vec<Example>,
    /// `data/rejected.jsonl`
    pub rejected: Vec<Rejected>,
}

impl Dataset {
    /// Reads the four files of `files`; a missing file reads as empty.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] when a file cannot be read or parsed.
    pub fn read(files: &DataFiles) -> Result<Self, DatasetError> {
        Ok(Self {
            subtopics: read(&files.subtopics)?,
            questions: read(&files.questions)?,
            answers: read(&files.answers)?,
            rejected: read(&files.rejected)?,
        })
    }

    /// Writes the files `change` touched, and appends its rejected records: every
    /// rewritten file is staged first, then the records are appended, then the
    /// files are renamed over the old ones in the order rejected, subtopics,
    /// questions, answers.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] when a file cannot be written; when staging fails,
    /// nothing changed.
    pub fn save(&self, files: &DataFiles, change: &Change) -> Result<(), DatasetError> {
        let mut rewrite = Rewrite::new();
        let touched = change.touched;
        if touched.contains(Touched::REJECTED) {
            let rejected: Vec<Rejected> = self
                .rejected
                .iter()
                .chain(&change.append)
                .cloned()
                .collect();
            rewrite.stage(&files.rejected, &rejected)?;
        }
        if touched.contains(Touched::SUBTOPICS) {
            rewrite.stage(&files.subtopics, &self.subtopics)?;
        }
        if touched.contains(Touched::QUESTIONS) {
            rewrite.stage(&files.questions, &self.questions)?;
        }
        if touched.contains(Touched::ANSWERS) {
            rewrite.stage(&files.answers, &self.answers)?;
        }
        if !touched.contains(Touched::REJECTED) && !change.append.is_empty() {
            let mut appender = super::Appender::open(&files.rejected)?;
            for record in &change.append {
                appender.append(record)?;
            }
        }
        rewrite.commit()
    }

    /// Replaces the text of question `id`, which must still read `before`.
    ///
    /// When the new text keeps the question's ID (only case or spacing changed),
    /// its answer keeps its ID and gets the new text as its user message. Otherwise
    /// the question gets its new ID and its answer is deleted: it answers another
    /// question.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::Changed`] when the question is gone or reads otherwise,
    /// and [`EditError::QuestionExists`] when the new ID is another question's.
    pub fn edit_question(
        &mut self,
        id: &Id,
        before: &str,
        text: &str,
    ) -> Result<Change, EditError> {
        let index = self
            .questions
            .iter()
            .position(|question| &question.id == id && question.text == before)
            .ok_or(EditError::Changed)?;
        let new_id = Id::question(&self.questions[index].subtopic_id, text);
        if &new_id == id {
            self.questions[index].text = text.to_string();
            let Some(example) = self.answers.iter_mut().find(|example| &example.id == id) else {
                return Ok(Change::new(Touched::QUESTIONS, "question saved"));
            };
            for message in &mut example.messages {
                if message.role == Role::User {
                    message.content = text.to_string();
                }
            }
            return Ok(Change::new(
                Touched::QUESTIONS.and(Touched::ANSWERS),
                "question saved",
            ));
        }
        if self.questions.iter().any(|question| question.id == new_id) {
            return Err(EditError::QuestionExists);
        }
        let question = &mut self.questions[index];
        question.id = new_id;
        question.text = text.to_string();
        let answers = self.answers.len();
        self.answers.retain(|example| &example.id != id);
        if self.answers.len() == answers {
            return Ok(Change::new(Touched::QUESTIONS, "question saved"));
        }
        Ok(Change::new(
            Touched::QUESTIONS.and(Touched::ANSWERS),
            "question saved; its old answer was deleted: 1 question unanswered, run answers to answer it",
        ))
    }

    /// Replaces the assistant message of answer `id`, which must still read
    /// `before`. The ID, the question and `meta` are left as they are.
    ///
    /// A reasoning that is `Some` but blank once trimmed is treated as absent.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::Changed`] when the answer is gone or reads otherwise,
    /// and [`EditError::ReasoningRemoved`] when `after` has no reasoning (blank
    /// included) but the answer has one.
    pub fn edit_answer(
        &mut self,
        id: &Id,
        before: &AnswerText,
        after: AnswerText,
    ) -> Result<Change, EditError> {
        let example = self
            .answers
            .iter_mut()
            .find(|example| &example.id == id && AnswerText::of(example).as_ref() == Some(before))
            .ok_or(EditError::Changed)?;
        let reasoning = after
            .reasoning
            .filter(|reasoning| !reasoning.trim().is_empty());
        if before.reasoning.is_some() && reasoning.is_none() {
            return Err(EditError::ReasoningRemoved);
        }
        if let Some(message) = example
            .messages
            .iter_mut()
            .find(|message| message.role == Role::Assistant)
        {
            message.content = after.content;
            message.reasoning_content = reasoning;
        }
        Ok(Change::new(Touched::ANSWERS, "answer saved"))
    }

    /// Renames subtopic `id`, which must still be named `before`, to the one-line
    /// `name`.
    ///
    /// When the new name keeps the subtopic's ID (only case or spacing changed),
    /// every copy of the name is replaced. Otherwise the subtopic gets its new ID
    /// and so does each of its questions (the question texts did not change), with
    /// their answers and their rejected questions re-keyed.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::MultiLine`] for a name of several lines,
    /// [`EditError::Changed`] when the subtopic is gone or named otherwise,
    /// [`EditError::SubtopicExists`] when the new ID is another subtopic's, and
    /// [`EditError::Rekey`] when a recomputed question ID is already used.
    pub fn rename_subtopic(
        &mut self,
        id: &Id,
        before: &str,
        name: &str,
    ) -> Result<Change, EditError> {
        if name.contains(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}']) {
            return Err(EditError::MultiLine);
        }
        let index = self
            .subtopics
            .iter()
            .position(|subtopic| &subtopic.id == id && subtopic.name == before)
            .ok_or(EditError::Changed)?;
        let topic = self.subtopics[index].topic.clone();
        let new_sid = Id::subtopic(&topic, name);
        if &new_sid == id {
            self.rename_in_place(index, name);
            return Ok(Change::new(
                Touched::SUBTOPICS
                    .and(Touched::QUESTIONS)
                    .and(Touched::ANSWERS),
                "subtopic renamed",
            ));
        }
        if self.subtopics.iter().any(|subtopic| subtopic.id == new_sid) {
            return Err(EditError::SubtopicExists { topic });
        }
        let keys: BTreeMap<Id, Id> = self
            .questions
            .iter()
            .filter(|question| &question.subtopic_id == id)
            .map(|question| (question.id.clone(), Id::question(&new_sid, &question.text)))
            .collect();
        let taken = self.taken(&keys);
        if taken > 0 {
            return Err(EditError::Rekey { count: taken });
        }
        Ok(self.rekey(index, &new_sid, name, &keys))
    }

    /// Replaces the name of subtopic `index` and every copy of it, IDs unchanged.
    fn rename_in_place(&mut self, index: usize, name: &str) {
        let subtopic = &mut self.subtopics[index];
        subtopic.name = name.to_string();
        let id = subtopic.id.clone();
        let mut questions = BTreeSet::new();
        for question in &mut self.questions {
            if question.subtopic_id == id {
                question.subtopic = name.to_string();
                questions.insert(question.id.clone());
            }
        }
        for example in &mut self.answers {
            if questions.contains(&example.id) {
                example.subtopic = name.to_string();
            }
        }
    }

    /// How many new IDs of `keys` are already used: by a question or an answer that
    /// `keys` does not re-key, or by another recomputed ID (two questions whose
    /// texts normalize the same).
    fn taken(&self, keys: &BTreeMap<Id, Id>) -> usize {
        let used: BTreeSet<&Id> = self
            .questions
            .iter()
            .map(|question| &question.id)
            .chain(self.answers.iter().map(|example| &example.id))
            .filter(|id| !keys.contains_key(*id))
            .collect();
        let mut recomputed: BTreeSet<&Id> = BTreeSet::new();
        keys.values()
            .filter(|new_id| used.contains(new_id) || !recomputed.insert(new_id))
            .count()
    }

    /// Gives subtopic `index` its new ID and name, and its questions, answers and
    /// rejected questions their new IDs from `keys`.
    fn rekey(&mut self, index: usize, new_sid: &Id, name: &str, keys: &BTreeMap<Id, Id>) -> Change {
        let old_sid = std::mem::replace(&mut self.subtopics[index].id, new_sid.clone());
        self.subtopics[index].name = name.to_string();
        for question in &mut self.questions {
            if let Some(new_id) = keys.get(&question.id) {
                question.id = new_id.clone();
                question.subtopic_id = new_sid.clone();
                question.subtopic = name.to_string();
            }
        }
        let mut answers = 0;
        for example in &mut self.answers {
            if let Some(new_id) = keys.get(&example.id) {
                example.id = new_id.clone();
                example.subtopic = name.to_string();
                answers += 1;
            }
        }
        let mut touched = Touched::SUBTOPICS
            .and(Touched::QUESTIONS)
            .and(Touched::ANSWERS);
        for record in &mut self.rejected {
            if let Rejected::Question {
                id,
                subtopic_id,
                text,
                ..
            } = record
                && *subtopic_id == old_sid
            {
                *id = Id::question(new_sid, text);
                subtopic_id.clone_from(new_sid);
                touched = touched.and(Touched::REJECTED);
            }
        }
        Change::new(
            touched,
            format!(
                "subtopic renamed; {} question ID{} recomputed, {answers} answer{} re-keyed",
                keys.len(),
                plural(keys.len()),
                plural(answers)
            ),
        )
    }
}

impl Dataset {
    /// The questions of `topic` whose subtopic is not in `data/subtopics.jsonl`.
    pub fn missing_subtopic<'a>(&'a self, topic: &'a str) -> impl Iterator<Item = &'a Question> {
        let subtopics: BTreeSet<&Id> = self.subtopics.iter().map(|subtopic| &subtopic.id).collect();
        self.questions.iter().filter(move |question| {
            question.topic == topic && !subtopics.contains(&question.subtopic_id)
        })
    }

    /// The questions `deletion` removes, subtopic and answer deletions aside.
    fn doomed_questions(&self, deletion: &Deletion) -> BTreeSet<Id> {
        match deletion {
            Deletion::Subtopic(id) => self
                .questions
                .iter()
                .filter(|question| &question.subtopic_id == id)
                .map(|question| question.id.clone())
                .collect(),
            Deletion::Question(id) => self
                .questions
                .iter()
                .filter(|question| &question.id == id)
                .map(|question| question.id.clone())
                .collect(),
            Deletion::Answer(_) => BTreeSet::new(),
            Deletion::MissingSubtopic(topic) => self
                .missing_subtopic(topic)
                .map(|question| question.id.clone())
                .collect(),
        }
    }

    /// Whether the item `deletion` names exists.
    fn exists(&self, deletion: &Deletion) -> bool {
        match deletion {
            Deletion::Subtopic(id) => self.subtopics.iter().any(|subtopic| &subtopic.id == id),
            Deletion::Question(id) => self.questions.iter().any(|question| &question.id == id),
            Deletion::Answer(id) => self.answers.iter().any(|example| &example.id == id),
            Deletion::MissingSubtopic(topic) => self.missing_subtopic(topic).next().is_some(),
        }
    }

    /// The IDs of the answers `deletion` removes: those of its questions, or the
    /// answer itself.
    fn doomed_answers(&self, deletion: &Deletion, questions: &BTreeSet<Id>) -> BTreeSet<Id> {
        self.answers
            .iter()
            .filter(|example| match deletion {
                Deletion::Answer(id) => &example.id == id,
                _ => questions.contains(&example.id),
            })
            .map(|example| example.id.clone())
            .collect()
    }

    /// What `deletion` removes, or `None` when what it names does not exist.
    #[must_use]
    pub fn counts(&self, deletion: &Deletion) -> Option<Counts> {
        if !self.exists(deletion) {
            return None;
        }
        let questions = self.doomed_questions(deletion);
        let answers = self.doomed_answers(deletion, &questions);
        Some(Counts {
            questions: questions.len(),
            answers: answers.len(),
        })
    }

    /// Applies `deletion`, which must still remove `expected`; the change records a
    /// deleted subtopic or question in `data/rejected.jsonl`.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::Changed`] when what `deletion` names is gone, and
    /// [`EditError::CountsChanged`] when it would remove other counts.
    pub fn delete(&mut self, deletion: &Deletion, expected: Counts) -> Result<Change, EditError> {
        let counts = self.counts(deletion).ok_or(EditError::Changed)?;
        if counts != expected {
            return Err(EditError::CountsChanged);
        }
        let questions = self.doomed_questions(deletion);
        let answers = self.doomed_answers(deletion, &questions);
        let append: Vec<Rejected> = self.rejection(deletion).into_iter().collect();
        let message = deleted(deletion, &append, counts);
        let mut touched = Touched::NONE;
        if let Deletion::Subtopic(id) = deletion {
            self.subtopics.retain(|subtopic| &subtopic.id != id);
            touched = Touched::SUBTOPICS;
        }
        if !questions.is_empty() {
            self.questions
                .retain(|question| !questions.contains(&question.id));
            touched = touched.and(Touched::QUESTIONS);
        }
        if !answers.is_empty() {
            self.answers
                .retain(|example| !answers.contains(&example.id));
            touched = touched.and(Touched::ANSWERS);
        }
        Ok(Change {
            touched,
            append,
            message,
        })
    }

    /// The record of a deleted subtopic or question for `data/rejected.jsonl`.
    fn rejection(&self, deletion: &Deletion) -> Option<Rejected> {
        match deletion {
            Deletion::Subtopic(id) => self
                .subtopics
                .iter()
                .find(|subtopic| &subtopic.id == id)
                .map(|subtopic| Rejected::Subtopic {
                    id: subtopic.id.clone(),
                    topic: subtopic.topic.clone(),
                    name: subtopic.name.clone(),
                }),
            Deletion::Question(id) => self
                .questions
                .iter()
                .find(|question| &question.id == id)
                .map(|question| Rejected::Question {
                    id: question.id.clone(),
                    topic: question.topic.clone(),
                    subtopic_id: question.subtopic_id.clone(),
                    text: question.text.clone(),
                }),
            Deletion::Answer(_) | Deletion::MissingSubtopic(_) => None,
        }
    }
}

/// What a deletion did, for the user; `rejected` holds its rejected record.
fn deleted(deletion: &Deletion, rejected: &[Rejected], counts: Counts) -> String {
    let Counts { questions, answers } = counts;
    match deletion {
        Deletion::Subtopic(_) => {
            let name = rejected
                .iter()
                .find_map(|record| match record {
                    Rejected::Subtopic { name, .. } => Some(name.as_str()),
                    Rejected::Question { .. } => None,
                })
                .unwrap_or_default();
            format!(
                "subtopic \"{name}\" deleted with its {questions} question{} and {answers} answer{}; \
                 recorded in data/rejected.jsonl",
                plural(questions),
                plural(answers)
            )
        },
        Deletion::Question(_) if answers > 0 => {
            "question deleted with its answer; recorded in data/rejected.jsonl".to_string()
        },
        Deletion::Question(_) => "question deleted; recorded in data/rejected.jsonl".to_string(),
        Deletion::Answer(_) => {
            "answer deleted; the next answers run asks the parent again".to_string()
        },
        Deletion::MissingSubtopic(_) => format!(
            "{questions} question{} whose subtopic no longer exists deleted, and their {answers} answer{}",
            plural(questions),
            plural(answers)
        ),
    }
}

/// `"s"` unless `count` is 1.
fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Exclusion, FinishReason, Message, Meta, ReasoningKind, Role};

    const TOPIC: &str = "ownership";
    const ANSWERED: &str = "What is a borrow?";
    const OPEN: &str = "Why borrow at all?";

    fn subtopic(name: &str) -> Subtopic {
        Subtopic {
            id: Id::subtopic(TOPIC, name),
            topic: TOPIC.into(),
            name: name.into(),
        }
    }

    fn question(subtopic: &Subtopic, text: &str) -> Question {
        Question {
            id: Id::question(&subtopic.id, text),
            topic: TOPIC.into(),
            subtopic_id: subtopic.id.clone(),
            subtopic: subtopic.name.clone(),
            text: text.into(),
        }
    }

    fn answer(question: &Question, reasoning: Option<&str>) -> Example {
        Example {
            id: question.id.clone(),
            topic: TOPIC.into(),
            subtopic: question.subtopic.clone(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: question.text.clone(),
                    reasoning_content: None,
                },
                Message {
                    role: Role::Assistant,
                    content: "Because.".into(),
                    reasoning_content: reasoning.map(str::to_string),
                },
            ],
            meta: Meta {
                model: "m".into(),
                input_tokens: 1,
                output_tokens: 2,
                finish_reason: FinishReason::Stop,
                reasoning_kind: ReasoningKind::Raw,
                excluded: None,
            },
        }
    }

    /// Borrowing with an answered and an open question, and an empty Lifetimes.
    fn dataset() -> Dataset {
        let borrowing = subtopic("Borrowing");
        let answered = question(&borrowing, ANSWERED);
        let open = question(&borrowing, OPEN);
        Dataset {
            answers: vec![answer(&answered, Some("Let me think."))],
            questions: vec![answered, open],
            subtopics: vec![borrowing, subtopic("Lifetimes")],
            rejected: Vec::new(),
        }
    }

    fn borrowing_id() -> Id {
        Id::subtopic(TOPIC, "Borrowing")
    }

    fn answered_id() -> Id {
        Id::question(&borrowing_id(), ANSWERED)
    }

    fn user_text(example: &Example) -> Option<&str> {
        example
            .messages
            .iter()
            .find(|message| message.role == Role::User)
            .map(|message| message.content.as_str())
    }

    #[test]
    fn a_question_edit_that_keeps_its_id_updates_the_answer_user_message() -> Result<(), EditError>
    {
        let mut data = dataset();
        let change = data.edit_question(&answered_id(), ANSWERED, "what is a  BORROW?")?;
        assert_eq!(data.questions[0].id, answered_id());
        assert_eq!(data.questions[0].text, "what is a  BORROW?");
        assert_eq!(data.answers[0].id, answered_id());
        assert_eq!(user_text(&data.answers[0]), Some("what is a  BORROW?"));
        assert_eq!(change.touched, Touched::QUESTIONS.and(Touched::ANSWERS));
        assert_eq!(change.message, "question saved");
        Ok(())
    }

    #[test]
    fn a_question_edit_with_a_new_id_deletes_its_old_answer() -> Result<(), EditError> {
        let mut data = dataset();
        let change = data.edit_question(&answered_id(), ANSWERED, "What is a shared borrow?")?;
        assert_eq!(
            data.questions[0].id,
            Id::question(&borrowing_id(), "What is a shared borrow?")
        );
        assert_eq!(data.questions[0].subtopic_id, borrowing_id());
        assert!(data.answers.is_empty());
        assert_eq!(change.touched, Touched::QUESTIONS.and(Touched::ANSWERS));
        assert_eq!(
            change.message,
            "question saved; its old answer was deleted: 1 question unanswered, run answers to answer it"
        );
        let open = Id::question(&borrowing_id(), OPEN);
        let change = data.edit_question(&open, OPEN, "Why borrow at all, then?")?;
        assert_eq!(change.touched, Touched::QUESTIONS);
        assert_eq!(change.message, "question saved");
        Ok(())
    }

    #[test]
    fn a_question_edit_onto_another_question_is_refused() {
        let mut data = dataset();
        let before = data.clone();
        let result = data.edit_question(&answered_id(), ANSWERED, "why BORROW at all?");
        assert!(matches!(result, Err(EditError::QuestionExists)));
        assert_eq!(data, before);
    }

    #[test]
    fn an_edit_of_something_changed_on_disk_is_refused() {
        let mut data = dataset();
        let before = data.clone();
        let stale = data.edit_question(&answered_id(), "What was a borrow?", "New?");
        assert!(matches!(stale, Err(EditError::Changed)));
        let gone = data.edit_question(&Id::of(&["gone"]), ANSWERED, "New?");
        assert!(matches!(gone, Err(EditError::Changed)));
        let renamed = data.rename_subtopic(&borrowing_id(), "Borrows", "Loans");
        assert!(matches!(renamed, Err(EditError::Changed)));
        let text = AnswerText {
            reasoning: None,
            content: "Because.".into(),
        };
        let answered = data.edit_answer(&answered_id(), &text, text.clone());
        assert!(matches!(answered, Err(EditError::Changed)));
        assert_eq!(data, before);
    }

    #[test]
    fn an_answer_edit_replaces_its_content_and_reasoning_only() -> Result<(), EditError> {
        let mut data = dataset();
        let meta = data.answers[0].meta.clone();
        let before = AnswerText {
            reasoning: Some("Let me think.".into()),
            content: "Because.".into(),
        };
        let after = AnswerText {
            reasoning: Some("Think again.".into()),
            content: "Because of moves.".into(),
        };
        let change = data.edit_answer(&answered_id(), &before, after.clone())?;
        assert_eq!(change.touched, Touched::ANSWERS);
        assert_eq!(change.message, "answer saved");
        let example = &data.answers[0];
        assert_eq!(example.id, answered_id());
        assert_eq!(example.meta, meta);
        assert_eq!(AnswerText::of(example), Some(after));
        assert_eq!(user_text(example), Some(ANSWERED));
        Ok(())
    }

    #[test]
    fn an_answer_edit_keeps_its_exclusion() -> Result<(), EditError> {
        let mut data = dataset();
        data.answers[0].meta.excluded = Some(Exclusion::Truncated);
        let meta = data.answers[0].meta.clone();
        let before = AnswerText {
            reasoning: Some("Let me think.".into()),
            content: "Because.".into(),
        };
        let after = AnswerText {
            reasoning: Some("Think again.".into()),
            content: "Because of moves.".into(),
        };
        data.edit_answer(&answered_id(), &before, after)?;
        assert_eq!(data.answers[0].meta, meta);
        Ok(())
    }

    #[test]
    fn emptying_an_existing_reasoning_with_only_whitespace_is_refused() {
        let mut data = dataset();
        let before_data = data.clone();
        let before = AnswerText {
            reasoning: Some("Let me think.".into()),
            content: "Because.".into(),
        };
        let blank = AnswerText {
            reasoning: Some("   ".into()),
            content: "Because.".into(),
        };
        let result = data.edit_answer(&answered_id(), &before, blank);
        assert!(matches!(result, Err(EditError::ReasoningRemoved)));
        assert_eq!(data, before_data);
    }

    #[test]
    fn emptying_an_existing_reasoning_is_refused_but_an_absent_one_stays_absent()
    -> Result<(), EditError> {
        let mut data = dataset();
        let before = AnswerText {
            reasoning: Some("Let me think.".into()),
            content: "Because.".into(),
        };
        let emptied = AnswerText {
            reasoning: None,
            content: "Because.".into(),
        };
        let result = data.edit_answer(&answered_id(), &before, emptied.clone());
        assert!(matches!(result, Err(EditError::ReasoningRemoved)));
        if let Some(message) = data.answers[0].messages.get_mut(1) {
            message.reasoning_content = None;
        }
        let edited = AnswerText {
            reasoning: None,
            content: "Because, really.".into(),
        };
        data.edit_answer(&answered_id(), &emptied, edited.clone())?;
        assert_eq!(AnswerText::of(&data.answers[0]), Some(edited.clone()));
        let blank = AnswerText {
            reasoning: Some("  ".into()),
            content: "Because, truly.".into(),
        };
        data.edit_answer(&answered_id(), &edited, blank)?;
        assert_eq!(
            AnswerText::of(&data.answers[0]),
            Some(AnswerText {
                reasoning: None,
                content: "Because, truly.".into(),
            })
        );
        Ok(())
    }

    #[test]
    fn a_rename_that_keeps_the_id_renames_every_copy_of_the_name() -> Result<(), EditError> {
        let mut data = dataset();
        let change = data.rename_subtopic(&borrowing_id(), "Borrowing", "BORROWING")?;
        assert_eq!(data.subtopics[0].id, borrowing_id());
        assert_eq!(data.subtopics[0].name, "BORROWING");
        assert!(data.questions.iter().all(|q| q.subtopic == "BORROWING"));
        assert_eq!(data.answers[0].subtopic, "BORROWING");
        assert_eq!(
            change.touched,
            Touched::SUBTOPICS
                .and(Touched::QUESTIONS)
                .and(Touched::ANSWERS)
        );
        assert_eq!(change.message, "subtopic renamed");
        Ok(())
    }

    #[test]
    fn a_rename_rekeys_questions_answers_and_rejections() -> Result<(), EditError> {
        let mut data = dataset();
        let rejected = "Is a borrow a pointer?";
        data.rejected.push(Rejected::Question {
            id: Id::question(&borrowing_id(), rejected),
            topic: TOPIC.into(),
            subtopic_id: borrowing_id(),
            text: rejected.into(),
        });
        let change = data.rename_subtopic(&borrowing_id(), "Borrowing", "References")?;
        let new_sid = Id::subtopic(TOPIC, "References");
        assert_eq!(
            new_sid.as_str(),
            Id::subtopic(TOPIC, " references ").as_str()
        );
        assert_eq!(data.subtopics[0].id, new_sid);
        assert_eq!(data.subtopics[0].name, "References");
        let answered = Id::question(&new_sid, ANSWERED);
        assert_eq!(data.questions[0].id, answered);
        assert_eq!(data.questions[0].subtopic_id, new_sid);
        assert_eq!(data.questions[1].id, Id::question(&new_sid, OPEN));
        assert_eq!(data.answers[0].id, answered);
        assert_eq!(data.answers[0].subtopic, "References");
        assert_eq!(user_text(&data.answers[0]), Some(ANSWERED));
        assert_eq!(data.answers[0].messages[1].content, "Because.");
        assert_eq!(
            data.rejected,
            [Rejected::Question {
                id: Id::question(&new_sid, rejected),
                topic: TOPIC.into(),
                subtopic_id: new_sid.clone(),
                text: rejected.into(),
            }]
        );
        assert_eq!(change.touched, Touched::ALL);
        assert_eq!(
            change.message,
            "subtopic renamed; 2 question IDs recomputed, 1 answer re-keyed"
        );
        Ok(())
    }

    #[test]
    fn a_rename_onto_an_existing_subtopic_or_a_leftover_id_is_refused() {
        let mut data = dataset();
        let before = data.clone();
        let taken = data.rename_subtopic(&borrowing_id(), "Borrowing", " lifetimes");
        assert!(
            matches!(taken, Err(EditError::SubtopicExists { ref topic }) if topic == TOPIC),
            "{taken:?}"
        );
        let multiline = data.rename_subtopic(&borrowing_id(), "Borrowing", "Two\nlines");
        assert!(matches!(multiline, Err(EditError::MultiLine)));
        // A leftover answer already holds the ID a recomputed question would get.
        let mut leftover = data.answers[0].clone();
        leftover.id = Id::question(&Id::subtopic(TOPIC, "References"), OPEN);
        data.answers.push(leftover);
        let with_leftover = data.clone();
        let rekey = data.rename_subtopic(&borrowing_id(), "Borrowing", "References");
        assert!(
            matches!(rekey, Err(EditError::Rekey { count: 1 })),
            "{rekey:?}"
        );
        assert_eq!(data, with_leftover);
        assert_ne!(data, before);
    }

    #[test]
    fn a_rename_refused_when_two_recomputed_question_ids_collide() {
        let mut data = dataset();
        data.questions.push(Question {
            id: Id::of(&["clash-one"]),
            topic: TOPIC.into(),
            subtopic_id: borrowing_id(),
            subtopic: "Borrowing".into(),
            text: "Why fear the borrow checker?".into(),
        });
        data.questions.push(Question {
            id: Id::of(&["clash-two"]),
            topic: TOPIC.into(),
            subtopic_id: borrowing_id(),
            subtopic: "Borrowing".into(),
            text: "why FEAR the borrow checker?".into(),
        });
        let before = data.clone();
        let rekey = data.rename_subtopic(&borrowing_id(), "Borrowing", "References");
        assert!(
            matches!(rekey, Err(EditError::Rekey { count: 1 })),
            "{rekey:?}"
        );
        assert_eq!(data, before);
    }

    #[test]
    fn a_rename_with_any_line_separator_is_refused_as_multiline() {
        let mut data = dataset();
        let before = data.clone();
        for name in ["A\rB", "A\u{85}B", "A\u{2028}B", "A\u{2029}B"] {
            let result = data.rename_subtopic(&borrowing_id(), "Borrowing", name);
            assert!(matches!(result, Err(EditError::MultiLine)), "{name:?}");
        }
        assert_eq!(data, before);
    }

    #[test]
    fn save_rewrites_the_touched_files_only() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let files = DataFiles::new(dir.path());
        let mut data = dataset();
        crate::dataset::rewrite(&files.subtopics, &data.subtopics)?;
        crate::dataset::rewrite(&files.questions, &data.questions)?;
        crate::dataset::rewrite(&files.answers, &data.answers)?;
        assert_eq!(Dataset::read(&files)?, data);
        let subtopics_before = std::fs::read(&files.subtopics)?;
        let change = data.edit_question(&answered_id(), ANSWERED, "What is a shared borrow?")?;
        data.save(&files, &change)?;
        assert_eq!(Dataset::read(&files)?, data);
        assert_eq!(std::fs::read(&files.subtopics)?, subtopics_before);
        assert!(!files.rejected.exists());
        Ok(())
    }

    #[test]
    fn save_stages_existing_and_appended_rejected_records_together()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let files = DataFiles::new(dir.path());
        let mut data = dataset();
        data.rejected.push(Rejected::Subtopic {
            id: Id::subtopic(TOPIC, "Deleted"),
            topic: TOPIC.into(),
            name: "Deleted".into(),
        });
        crate::dataset::rewrite(&files.rejected, &data.rejected)?;
        let appended = Rejected::Question {
            id: Id::question(&borrowing_id(), "Is a borrow free?"),
            topic: TOPIC.into(),
            subtopic_id: borrowing_id(),
            text: "Is a borrow free?".into(),
        };
        let change = Change {
            touched: Touched::REJECTED,
            append: vec![appended.clone()],
            message: "test".into(),
        };
        data.save(&files, &change)?;
        let mut expected = data.rejected.clone();
        expected.push(appended);
        assert_eq!(read::<Rejected>(&files.rejected)?, expected);
        Ok(())
    }

    #[test]
    fn counts_say_what_a_deletion_removes() {
        let mut data = dataset();
        let open = Id::question(&borrowing_id(), OPEN);
        let counts = |questions, answers| Some(Counts { questions, answers });
        assert_eq!(
            data.counts(&Deletion::Subtopic(borrowing_id())),
            counts(2, 1)
        );
        assert_eq!(
            data.counts(&Deletion::Question(answered_id())),
            counts(1, 1)
        );
        assert_eq!(data.counts(&Deletion::Question(open)), counts(1, 0));
        assert_eq!(data.counts(&Deletion::Answer(answered_id())), counts(0, 1));
        assert_eq!(data.counts(&Deletion::Answer(Id::of(&["none"]))), None);
        assert_eq!(data.counts(&Deletion::MissingSubtopic(TOPIC.into())), None);
        data.subtopics.remove(0);
        assert_eq!(
            data.counts(&Deletion::MissingSubtopic(TOPIC.into())),
            counts(2, 1)
        );
        assert_eq!(data.missing_subtopic(TOPIC).count(), 2);
    }

    #[test]
    fn deleting_a_subtopic_cascades_and_records_only_the_subtopic() -> Result<(), EditError> {
        let mut data = dataset();
        let expected = Counts {
            questions: 2,
            answers: 1,
        };
        let change = data.delete(&Deletion::Subtopic(borrowing_id()), expected)?;
        assert_eq!(data.subtopics, [subtopic("Lifetimes")]);
        assert!(data.questions.is_empty());
        assert!(data.answers.is_empty());
        assert_eq!(
            change.append,
            [Rejected::Subtopic {
                id: borrowing_id(),
                topic: TOPIC.into(),
                name: "Borrowing".into(),
            }]
        );
        assert_eq!(
            change.touched,
            Touched::SUBTOPICS
                .and(Touched::QUESTIONS)
                .and(Touched::ANSWERS)
        );
        assert_eq!(
            change.message,
            "subtopic \"Borrowing\" deleted with its 2 questions and 1 answer; recorded in data/rejected.jsonl"
        );
        Ok(())
    }

    #[test]
    fn deleting_a_question_takes_its_answer_and_records_the_question() -> Result<(), EditError> {
        let mut data = dataset();
        let expected = Counts {
            questions: 1,
            answers: 1,
        };
        let change = data.delete(&Deletion::Question(answered_id()), expected)?;
        assert_eq!(data.questions.len(), 1);
        assert!(data.answers.is_empty());
        assert_eq!(
            change.append,
            [Rejected::Question {
                id: answered_id(),
                topic: TOPIC.into(),
                subtopic_id: borrowing_id(),
                text: ANSWERED.into(),
            }]
        );
        assert_eq!(change.touched, Touched::QUESTIONS.and(Touched::ANSWERS));
        assert_eq!(
            change.message,
            "question deleted with its answer; recorded in data/rejected.jsonl"
        );
        let open = Id::question(&borrowing_id(), OPEN);
        let expected = Counts {
            questions: 1,
            answers: 0,
        };
        let change = data.delete(&Deletion::Question(open), expected)?;
        assert_eq!(change.touched, Touched::QUESTIONS);
        assert_eq!(
            change.message,
            "question deleted; recorded in data/rejected.jsonl"
        );
        Ok(())
    }

    #[test]
    fn deleting_an_answer_keeps_its_question_and_records_nothing() -> Result<(), EditError> {
        let mut data = dataset();
        let expected = Counts {
            questions: 0,
            answers: 1,
        };
        let change = data.delete(&Deletion::Answer(answered_id()), expected)?;
        assert_eq!(data.questions.len(), 2);
        assert!(data.answers.is_empty());
        assert!(change.append.is_empty());
        assert_eq!(change.touched, Touched::ANSWERS);
        assert_eq!(
            change.message,
            "answer deleted; the next answers run asks the parent again"
        );
        Ok(())
    }

    #[test]
    fn deleting_the_missing_subtopic_group_records_nothing() -> Result<(), EditError> {
        let mut data = dataset();
        data.subtopics.remove(0);
        let expected = Counts {
            questions: 2,
            answers: 1,
        };
        let change = data.delete(&Deletion::MissingSubtopic(TOPIC.into()), expected)?;
        assert!(data.questions.is_empty());
        assert!(data.answers.is_empty());
        assert!(change.append.is_empty());
        assert_eq!(
            change.message,
            "2 questions whose subtopic no longer exists deleted, and their 1 answer"
        );
        Ok(())
    }

    #[test]
    fn a_deletion_whose_counts_changed_on_disk_is_refused() {
        let mut data = dataset();
        let before = data.clone();
        let stale = Counts {
            questions: 3,
            answers: 1,
        };
        let result = data.delete(&Deletion::Subtopic(borrowing_id()), stale);
        assert!(matches!(result, Err(EditError::CountsChanged)));
        let gone = data.delete(&Deletion::Answer(Id::of(&["none"])), Counts::default());
        assert!(matches!(gone, Err(EditError::Changed)));
        assert_eq!(data, before);
    }

    #[test]
    fn save_appends_the_rejected_records() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let files = DataFiles::new(dir.path());
        let mut data = dataset();
        data.save(&DataFiles::new(dir.path()), &Change::new(Touched::ALL, ""))?;
        let expected = Counts {
            questions: 1,
            answers: 1,
        };
        let change = data.delete(&Deletion::Question(answered_id()), expected)?;
        data.save(&files, &change)?;
        data.rejected.extend(change.append);
        assert_eq!(Dataset::read(&files)?, data);
        Ok(())
    }
}
