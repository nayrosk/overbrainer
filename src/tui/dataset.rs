//! The Dataset view's model, built once per load, filter change or edit: the tree
//! of topics, subtopics, questions and answers, and the stats of each topic.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use ratatui::text::Line;
use tui_tree_widget::{TreeItem, TreeState};

use crate::dataset::{Dataset, Example, Exclusion, Id, Question, normalize};
use crate::pipeline::{SplitClass, SplitReport, eval_size, split_class};

/// A topic of `overbrainer.toml`, as the Dataset view shows it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TopicInfo {
    /// Its name.
    pub(super) name: String,
    /// Its description.
    pub(super) description: Option<String>,
    /// Subtopics to generate.
    pub(super) subtopics: u32,
    /// Questions to generate per subtopic.
    pub(super) questions_per_subtopic: u32,
}

/// A node of the tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Node {
    /// A topic, configured or only found in the data.
    Topic(String),
    /// A subtopic.
    Subtopic(Id),
    /// The questions of a topic whose subtopic no longer exists.
    MissingSubtopic(String),
    /// A question.
    Question(Id),
    /// The answer of a question.
    Answer(Id),
}

/// The tree and stats of one loaded dataset.
pub(super) struct Model {
    /// The records, as read.
    pub(super) data: Dataset,
    /// The configured topic names.
    configured: BTreeSet<String>,
    /// Every topic: the configured ones in order, then the others by name.
    pub(super) topics: Vec<String>,
    /// The tree, or why it cannot be built (duplicate IDs in a hand-edited file).
    pub(super) items: Result<Vec<TreeItem<'static, Node>>, String>,
    /// Subtopics and questions matching the filter, when one applies.
    pub(super) matches: Option<usize>,
    /// Stats of each topic, and of all topics under `None`.
    pub(super) stats: BTreeMap<Option<String>, Stats>,
}

/// Counts and sizes of one topic, or of all topics.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Stats {
    /// Subtopics stored.
    pub(super) subtopics: usize,
    /// Subtopics configured, for configured topics.
    pub(super) subtopics_target: Option<u64>,
    /// Questions stored.
    pub(super) questions: usize,
    /// Questions configured, for configured topics.
    pub(super) questions_target: Option<u64>,
    /// Answers stored.
    pub(super) answers: usize,
    /// Questions without an answer.
    pub(super) unanswered: usize,
    /// Answers `split` uses.
    pub(super) usable: usize,
    /// Answers `split` leaves out, by reason.
    pub(super) excluded: BTreeMap<Exclusion, usize>,
    /// Answers `split` counts as orphaned.
    pub(super) orphaned: usize,
    /// Lengths of the question texts, in characters.
    pub(super) question_chars: Lengths,
    /// Lengths of the answers' contents.
    pub(super) answer_chars: Lengths,
    /// Lengths of the answers' reasonings, when they have one.
    pub(super) reasoning_chars: Lengths,
    /// Prompt tokens of the answers.
    pub(super) input_tokens: u64,
    /// Completion tokens of the answers.
    pub(super) output_tokens: u64,
    /// Entries of `data/rejected.jsonl`.
    pub(super) rejected: usize,
}

/// Mean and maximum of some lengths.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Lengths {
    count: usize,
    total: usize,
    /// The longest.
    pub(super) max: usize,
}

impl Lengths {
    fn add(&mut self, text: &str) {
        let length = text.chars().count();
        self.count += 1;
        self.total += length;
        self.max = self.max.max(length);
    }

    /// The mean, rounded down; 0 when empty.
    pub(super) fn mean(self) -> usize {
        self.total.checked_div(self.count).unwrap_or(0)
    }
}

impl Model {
    /// The model of `data` for the configured `topics`: stats now, and the tree
    /// filtered by `filter`.
    pub(super) fn new(data: Dataset, topics: &[TopicInfo], filter: &str) -> Self {
        let configured: BTreeSet<String> = topics.iter().map(|topic| topic.name.clone()).collect();
        let found: BTreeSet<&str> = data
            .subtopics
            .iter()
            .map(|subtopic| subtopic.topic.as_str())
            .chain(
                data.questions
                    .iter()
                    .map(|question| question.topic.as_str()),
            )
            .chain(data.answers.iter().map(|example| example.topic.as_str()))
            .filter(|name| !configured.contains(*name))
            .collect();
        let names: Vec<String> = topics
            .iter()
            .map(|topic| topic.name.clone())
            .chain(found.into_iter().map(str::to_string))
            .collect();
        let mut model = Self {
            data,
            configured,
            topics: names,
            items: Ok(Vec::new()),
            matches: None,
            stats: BTreeMap::new(),
        };
        model.stats = model.all_stats(topics);
        model.filter(filter);
        model
    }

    /// Rebuilds the tree for `filter`: a case-insensitive substring of question
    /// texts and subtopic names. A matching subtopic keeps all its questions, a
    /// matching question its parents.
    pub(super) fn filter(&mut self, filter: &str) {
        let needle = normalize(filter);
        let tree = Tree::new(self, &needle);
        let items: Result<Vec<_>, _> = self
            .topics
            .iter()
            .filter_map(|topic| tree.topic(topic).transpose())
            .collect();
        let matches = tree.matches.get();
        self.items = items.map_err(|error| format!("cannot show the dataset tree: {error}"));
        self.matches = (!needle.is_empty()).then_some(matches);
    }

    /// Whether `topic` is in `overbrainer.toml`.
    pub(super) fn is_configured(&self, topic: &str) -> bool {
        self.configured.contains(topic)
    }

    /// The answer of question `id`.
    pub(super) fn answer(&self, id: &Id) -> Option<&Example> {
        self.data.answers.iter().find(|example| &example.id == id)
    }

    /// How `split` treats `example`.
    pub(super) fn class(&self, example: &Example) -> SplitClass {
        let configured: BTreeSet<&str> = self.configured.iter().map(String::as_str).collect();
        let known: BTreeSet<&Id> = self.data.questions.iter().map(|q| &q.id).collect();
        split_class(&configured, &known, example)
    }

    fn all_stats(&self, topics: &[TopicInfo]) -> BTreeMap<Option<String>, Stats> {
        let configured: BTreeSet<&str> = self.configured.iter().map(String::as_str).collect();
        let known: BTreeSet<&Id> = self.data.questions.iter().map(|q| &q.id).collect();
        let answered: BTreeSet<&Id> = self.data.answers.iter().map(|e| &e.id).collect();
        let mut stats: BTreeMap<Option<String>, Stats> = BTreeMap::new();
        for topic in topics {
            let entry = stats.entry(Some(topic.name.clone())).or_default();
            entry.subtopics_target = Some(u64::from(topic.subtopics));
            entry.questions_target =
                Some(u64::from(topic.subtopics) * u64::from(topic.questions_per_subtopic));
        }
        let mut both = |topic: &str, add: &dyn Fn(&mut Stats)| {
            add(stats.entry(Some(topic.to_string())).or_default());
            add(stats.entry(None).or_default());
        };
        for subtopic in &self.data.subtopics {
            both(&subtopic.topic, &|s| s.subtopics += 1);
        }
        for question in &self.data.questions {
            let open = !answered.contains(&question.id);
            both(&question.topic, &|s| {
                s.questions += 1;
                s.unanswered += usize::from(open);
                s.question_chars.add(&question.text);
            });
        }
        for example in &self.data.answers {
            let class = split_class(&configured, &known, example);
            both(&example.topic, &|s| count_answer(s, example, class));
        }
        for record in &self.data.rejected {
            both(record.topic(), &|s| s.rejected += 1);
        }
        let all = stats.entry(None).or_default();
        all.subtopics_target = topics
            .iter()
            .map(|topic| u64::from(topic.subtopics))
            .reduce(|a, b| a + b);
        all.questions_target = topics
            .iter()
            .map(|t| u64::from(t.subtopics) * u64::from(t.questions_per_subtopic))
            .reduce(|a, b| a + b);
        stats
    }
}

fn count_answer(stats: &mut Stats, example: &Example, class: SplitClass) {
    stats.answers += 1;
    match class {
        SplitClass::Usable => stats.usable += 1,
        SplitClass::Excluded(reason) => *stats.excluded.entry(reason).or_default() += 1,
        SplitClass::Orphaned => stats.orphaned += 1,
    }
    if let Some(text) = crate::dataset::AnswerText::of(example) {
        stats.answer_chars.add(&text.content);
        if let Some(reasoning) = &text.reasoning {
            stats.reasoning_chars.add(reasoning);
        }
    }
    stats.input_tokens += example.meta.input_tokens;
    stats.output_tokens += example.meta.output_tokens;
}

/// The expected train and eval sizes of `usable` examples, or the sizes of the
/// last split, with a label saying which.
pub(super) fn sizes(
    usable: usize,
    ratio: f64,
    last: Option<&SplitReport>,
) -> (usize, usize, &'static str) {
    last.map_or_else(
        || {
            let eval = eval_size(usable, ratio);
            (usable - eval, eval, "expected")
        },
        |report| (report.train, report.eval, "last split"),
    )
}

/// The tree builder of one filter.
struct Tree<'a> {
    model: &'a Model,
    needle: &'a str,
    answers: HashMap<&'a Id, &'a Example>,
    subtopic_ids: BTreeSet<&'a Id>,
    matches: std::cell::Cell<usize>,
}

impl<'a> Tree<'a> {
    fn new(model: &'a Model, needle: &'a str) -> Self {
        Self {
            model,
            needle,
            answers: model.data.answers.iter().map(|e| (&e.id, e)).collect(),
            subtopic_ids: model.data.subtopics.iter().map(|s| &s.id).collect(),
            matches: std::cell::Cell::new(0),
        }
    }

    fn hit(&self, text: &str) -> bool {
        let hit = !self.needle.is_empty() && normalize(text).contains(self.needle);
        if hit {
            self.matches.set(self.matches.get() + 1);
        }
        hit
    }

    /// The node of `topic`, `None` when the filter leaves nothing of it.
    fn topic(&self, topic: &str) -> std::io::Result<Option<TreeItem<'static, Node>>> {
        let data = &self.model.data;
        let mut children = Vec::new();
        for subtopic in data.subtopics.iter().filter(|s| s.topic == topic) {
            let questions: Vec<&Question> = data
                .questions
                .iter()
                .filter(|q| q.subtopic_id == subtopic.id)
                .collect();
            if let Some(item) = self.group(
                Node::Subtopic(subtopic.id.clone()),
                &subtopic.name,
                self.hit(&subtopic.name),
                &questions,
            )? {
                children.push(item);
            }
        }
        let missing: Vec<&Question> = data
            .questions
            .iter()
            .filter(|q| q.topic == topic && !self.subtopic_ids.contains(&q.subtopic_id))
            .collect();
        if !missing.is_empty()
            && let Some(item) = self.group(
                Node::MissingSubtopic(topic.to_string()),
                "(missing subtopic)",
                false,
                &missing,
            )?
        {
            children.push(item);
        }
        if children.is_empty() && !self.needle.is_empty() {
            return Ok(None);
        }
        let label = self.topic_label(topic);
        TreeItem::new(Node::Topic(topic.to_string()), label, children).map(Some)
    }

    fn topic_label(&self, topic: &str) -> Line<'static> {
        let data = &self.model.data;
        let subtopics = data.subtopics.iter().filter(|s| s.topic == topic).count();
        let questions = data.questions.iter().filter(|q| q.topic == topic).count();
        let answers = data.answers.iter().filter(|e| e.topic == topic).count();
        let name = if self.model.is_configured(topic) {
            topic.to_string()
        } else {
            format!("(not configured) {topic}")
        };
        Line::from(format!(
            "{name}  {subtopics} sub, {questions} q, {answers} a"
        ))
    }

    /// A subtopic or the missing-subtopic group, `None` when the filter leaves
    /// nothing of it. A matching group keeps every question.
    fn group(
        &self,
        node: Node,
        name: &str,
        matched: bool,
        questions: &[&Question],
    ) -> std::io::Result<Option<TreeItem<'static, Node>>> {
        let mut children = Vec::new();
        for question in questions {
            if self.needle.is_empty() || matched || self.hit(&question.text) {
                children.push(self.question(question)?);
            }
        }
        if children.is_empty() && !self.needle.is_empty() && !matched {
            return Ok(None);
        }
        let answers = questions
            .iter()
            .filter(|q| self.answers.contains_key(&q.id))
            .count();
        let label = format!("{name}  {} q, {answers} a", questions.len());
        TreeItem::new(node, Line::from(label), children).map(Some)
    }

    fn question(&self, question: &Question) -> std::io::Result<TreeItem<'static, Node>> {
        let text = question
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let Some(example) = self.answers.get(&question.id) else {
            return Ok(TreeItem::new_leaf(
                Node::Question(question.id.clone()),
                format!("[ ] {text}"),
            ));
        };
        let (marker, note) = match self.model.class(example) {
            SplitClass::Usable => ("[a]", String::new()),
            SplitClass::Excluded(reason) => ("[x]", format!("  excluded: {}", exclusion(reason))),
            SplitClass::Orphaned => ("[o]", "  orphaned".to_string()),
        };
        let answer = TreeItem::new_leaf(Node::Answer(question.id.clone()), format!("answer{note}"));
        TreeItem::new(
            Node::Question(question.id.clone()),
            format!("{marker} {text}"),
            vec![answer],
        )
    }
}

/// The name of an exclusion, as `answers.jsonl` spells it.
pub(super) fn exclusion(reason: Exclusion) -> &'static str {
    match reason {
        Exclusion::Truncated => "truncated",
        Exclusion::Empty => "empty",
        Exclusion::Refused => "refused",
        Exclusion::NoRawReasoning => "no_raw_reasoning",
    }
}

/// The Dataset view's state.
#[derive(Default)]
pub(super) struct DatasetView {
    /// The loaded model; `None` until the first load ends.
    pub(super) model: Option<Model>,
    /// Why the last load failed.
    pub(super) error: Option<String>,
    /// The tree's selection and open nodes.
    pub(super) tree: TreeState<Node>,
    /// The applied filter.
    pub(super) filter: String,
    /// The filter being typed, while `/` is active.
    pub(super) input: Option<String>,
    /// Whether the stats pane replaces the detail pane.
    pub(super) stats: bool,
    /// Scroll of the detail pane, in lines.
    pub(super) scroll: u16,
    /// The last split run in this session.
    pub(super) split: Option<SplitReport>,
}

impl DatasetView {
    /// Shows a newly loaded `data`, keeping the selection and open nodes when the
    /// selected node is still shown, else selecting the first node.
    pub(super) fn loaded(&mut self, data: Dataset, topics: &[TopicInfo]) {
        self.model = Some(Model::new(data, topics, &self.filter));
        self.error = None;
        if !self.shown(self.tree.selected()) {
            self.tree.select(Vec::new());
            self.step(true);
        }
    }

    /// Whether the node at `path` is shown in the tree.
    fn shown(&self, path: &[Node]) -> bool {
        let Some(Ok(items)) = self.model.as_ref().map(|model| model.items.as_ref()) else {
            return false;
        };
        self.tree
            .flatten(items)
            .iter()
            .any(|flat| flat.identifier.as_slice() == path)
    }

    /// Moves the selection one visible node down, or up.
    pub(super) fn step(&mut self, down: bool) {
        let Some(Ok(items)) = self.model.as_ref().map(|model| model.items.as_ref()) else {
            return;
        };
        let visible = self.tree.flatten(items);
        let current = visible
            .iter()
            .position(|flat| flat.identifier.as_slice() == self.tree.selected());
        let next = match (current, down) {
            (None, _) => 0,
            (Some(index), true) => (index + 1).min(visible.len().saturating_sub(1)),
            (Some(index), false) => index.saturating_sub(1),
        };
        if let Some(flat) = visible.get(next) {
            let identifier = flat.identifier.clone();
            self.tree.select(identifier);
            self.scroll = 0;
        }
    }

    /// Applies `filter` and rebuilds the tree.
    pub(super) fn apply_filter(&mut self, filter: String) {
        self.filter = filter;
        if let Some(model) = &mut self.model {
            model.filter(&self.filter);
        }
        self.tree.select(Vec::new());
        self.step(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::snapshots::{MOVED, dataset, path_to, topics};

    /// The labels of the visible nodes, indented by depth.
    fn labels(view: &DatasetView) -> Vec<String> {
        let Some(Ok(items)) = view.model.as_ref().map(|model| model.items.as_ref()) else {
            return Vec::new();
        };
        view.tree
            .flatten(items)
            .iter()
            .map(|flat| format!("{}{:?}", "  ".repeat(flat.depth()), flat.identifier.last()))
            .collect()
    }

    fn loaded() -> DatasetView {
        let mut view = DatasetView::default();
        view.loaded(dataset(), &topics());
        view
    }

    #[test]
    fn topics_come_in_config_order_then_unconfigured_ones() {
        let view = loaded();
        let model = view.model.as_ref();
        assert_eq!(
            model.map(|m| m.topics.clone()),
            Some(vec![
                "ownership".to_string(),
                "traits".to_string(),
                "old_topic".to_string()
            ])
        );
        assert_eq!(view.tree.selected(), [Node::Topic("ownership".into())]);
        assert_eq!(labels(&view).len(), 3);
    }

    #[test]
    fn questions_group_under_their_subtopic_or_the_missing_one() {
        let mut view = loaded();
        view.tree.open(vec![Node::Topic("ownership".into())]);
        let shown = labels(&view);
        assert_eq!(shown.len(), 6, "{shown:#?}");
        assert!(
            shown[3].contains("MissingSubtopic(\"ownership\")"),
            "{shown:#?}"
        );
    }

    #[test]
    fn stepping_follows_the_visible_nodes() {
        let mut view = loaded();
        view.step(true);
        assert_eq!(view.tree.selected(), [Node::Topic("traits".into())]);
        view.step(true);
        view.step(true);
        assert_eq!(view.tree.selected(), [Node::Topic("old_topic".into())]);
        view.step(false);
        assert_eq!(view.tree.selected(), [Node::Topic("traits".into())]);
    }

    #[test]
    fn a_filter_keeps_matching_questions_with_their_parents() {
        let mut view = loaded();
        view.apply_filter("nll".into());
        let model = view.model.as_ref();
        assert_eq!(model.and_then(|m| m.matches), Some(1));
        view.tree.open(vec![Node::Topic("ownership".into())]);
        let path = path_to("When does NLL end a borrow?", false);
        view.tree.open(path[..2].to_vec());
        let shown = labels(&view);
        assert_eq!(shown.len(), 3, "{shown:#?}");
        view.apply_filter("no such text".into());
        assert_eq!(labels(&view), Vec::<String>::new());
        view.apply_filter(String::new());
        assert_eq!(view.model.as_ref().and_then(|m| m.matches), None);
        assert_eq!(labels(&view).len(), 9, "the open nodes stay open");
    }

    #[test]
    fn stats_follow_split_rules_per_topic_and_overall() {
        let view = loaded();
        let Some(model) = view.model.as_ref() else {
            return;
        };
        let ownership = model
            .stats
            .get(&Some("ownership".into()))
            .cloned()
            .unwrap_or_default();
        assert_eq!(
            (ownership.subtopics, ownership.questions, ownership.answers),
            (2, 5, 4)
        );
        assert_eq!(
            (ownership.usable, ownership.orphaned, ownership.unanswered),
            (3, 0, 1)
        );
        assert_eq!(ownership.excluded.get(&Exclusion::Truncated), Some(&1));
        assert_eq!(ownership.questions_target, Some(6));
        assert_eq!(ownership.rejected, 1);
        assert_eq!(ownership.question_chars.max, MOVED.chars().count());
        let old = model
            .stats
            .get(&Some("old_topic".into()))
            .cloned()
            .unwrap_or_default();
        assert_eq!((old.orphaned, old.subtopics_target), (1, None));
        let all = model.stats.get(&None).cloned().unwrap_or_default();
        assert_eq!((all.answers, all.usable, all.orphaned), (5, 3, 1));
        assert_eq!(all.subtopics_target, Some(4));
        assert_eq!(sizes(all.usable, 0.1, None), (2, 1, "expected"));
    }

    #[test]
    fn duplicate_ids_in_a_file_show_an_error_instead_of_the_tree() {
        let mut data = dataset();
        let copy = data.questions[0].clone();
        data.questions.push(copy);
        let mut view = DatasetView::default();
        view.loaded(data, &topics());
        let Some(model) = view.model.as_ref() else {
            return;
        };
        assert!(
            model
                .items
                .as_ref()
                .is_err_and(|error| error.starts_with("cannot show the dataset tree")),
        );
    }
}
