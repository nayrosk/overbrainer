//! The Dataset view's model, built once per load, filter change or edit: the tree
//! of topics, subtopics, questions and answers, and the stats of each topic.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use tui_tree_widget::{TreeItem, TreeState};

use crate::dataset::{Dataset, Example, Exclusion, Id, Question, Subtopic, normalize};
use crate::pipeline::{SplitClass, SplitReport, eval_size, split_class};

/// A node of the tree with its children.
type Item = TreeItem<'static, Node>;

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
    pub(super) items: Result<Vec<Item>, String>,
    /// Subtopics and questions matching the filter, when one applies.
    pub(super) matches: Option<usize>,
    /// The nodes to open so the filter's matches show: every topic left by the
    /// filter, and every group holding a matching question.
    pub(super) open: Vec<Vec<Node>>,
    /// Index in `data.answers` of the first answer of each question ID: the one
    /// the tree and the detail pane both show.
    first_answer: HashMap<Id, usize>,
    /// Index in `data.questions` of the first question of each ID.
    first_question: HashMap<Id, usize>,
    /// How `split` treats each answer, in the order of `data.answers`.
    classes: Vec<SplitClass>,
    /// Stats of each topic, and of all topics under `None`.
    pub(super) stats: BTreeMap<Option<String>, Stats>,
    /// The style of the `(not configured)` label of a topic.
    warn: Style,
    /// The filter the tree was built for, normalized.
    needle: String,
    /// The columns a topic's label gets in the tree, once drawn: its counts
    /// are left out when they do not fit whole.
    room: Option<usize>,
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
    /// filtered by `filter`, a topic not configured labelled in `warn`.
    pub(super) fn new(
        data: Dataset,
        topics: &[TopicInfo],
        (filter, room): (&str, Option<usize>),
        warn: Style,
    ) -> Self {
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
        let classes = {
            let names: BTreeSet<&str> = configured.iter().map(String::as_str).collect();
            let known: BTreeSet<&Id> = data.questions.iter().map(|q| &q.id).collect();
            data.answers
                .iter()
                .map(|example| split_class(&names, &known, example))
                .collect()
        };
        let mut first_answer = HashMap::new();
        for (index, example) in data.answers.iter().enumerate() {
            first_answer.entry(example.id.clone()).or_insert(index);
        }
        let mut first_question = HashMap::new();
        for (index, question) in data.questions.iter().enumerate() {
            first_question.entry(question.id.clone()).or_insert(index);
        }
        let mut model = Self {
            data,
            configured,
            topics: names,
            items: Ok(Vec::new()),
            matches: None,
            open: Vec::new(),
            first_answer,
            first_question,
            classes,
            stats: BTreeMap::new(),
            warn,
            needle: String::new(),
            room,
        };
        model.stats = model.all_stats(topics);
        model.filter(filter);
        model
    }

    /// Rebuilds the tree for `filter`: a case-insensitive substring of question
    /// texts and subtopic names. A matching subtopic keeps all its questions, a
    /// matching question its parents.
    pub(super) fn filter(&mut self, filter: &str) {
        self.needle = normalize(filter);
        let (items, matches, open) = self.build();
        self.items = items;
        self.matches = (!self.needle.is_empty()).then_some(matches);
        self.open = open;
    }

    /// Gives a topic's label `room` columns: the tree is built again when
    /// they changed, each topic's counts left out when they do not fit.
    pub(super) fn fit(&mut self, room: usize) {
        if self.room != Some(room) {
            self.room = Some(room);
            self.items = self.build().0;
        }
    }

    /// The tree for the filter, the matches it counts and the nodes to open
    /// so they show.
    fn build(&self) -> (Result<Vec<Item>, String>, usize, Vec<Vec<Node>>) {
        let tree = Tree::new(self, &self.needle);
        let items: Result<Vec<_>, _> = self
            .topics
            .iter()
            .filter_map(|topic| tree.topic(topic).transpose())
            .collect();
        let items = items.map_err(|error| format!("cannot show the dataset tree: {error}"));
        (items, tree.matches.get(), tree.open.take())
    }

    /// Whether `topic` is in `overbrainer.toml`.
    pub(super) fn is_configured(&self, topic: &str) -> bool {
        self.configured.contains(topic)
    }

    /// The answer of question `id`: the first one, when a hand-edited file holds
    /// several.
    pub(super) fn answer(&self, id: &Id) -> Option<&Example> {
        let index = *self.first_answer.get(id)?;
        self.data.answers.get(index)
    }

    /// How `split` treats the answer of question `id`, the one [`Model::answer`]
    /// gives; `None` when it has none.
    pub(super) fn class(&self, id: &Id) -> Option<SplitClass> {
        let index = *self.first_answer.get(id)?;
        self.classes.get(index).copied()
    }

    /// The question `id`: the first one, when a hand-edited file holds several.
    pub(super) fn question(&self, id: &Id) -> Option<&Question> {
        let index = *self.first_question.get(id)?;
        self.data.questions.get(index)
    }

    fn all_stats(&self, topics: &[TopicInfo]) -> BTreeMap<Option<String>, Stats> {
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
            let open = !self.first_answer.contains_key(&question.id);
            both(&question.topic, &|s| {
                s.questions += 1;
                s.unanswered += usize::from(open);
                s.question_chars.add(&question.text);
            });
        }
        for (example, class) in self.data.answers.iter().zip(&self.classes) {
            both(&example.topic, &|s| count_answer(s, example, *class));
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

/// The tree builder of one filter. Its maps group the records once, so a build
/// costs one pass over the data plus the hash lookups.
struct Tree<'a> {
    model: &'a Model,
    needle: &'a str,
    /// The subtopics of each topic, in file order.
    subtopics: HashMap<&'a str, Vec<&'a Subtopic>>,
    /// The questions of each subtopic ID, in file order.
    questions: HashMap<&'a Id, Vec<&'a Question>>,
    /// The questions of each topic whose subtopic is gone.
    missing: HashMap<&'a str, Vec<&'a Question>>,
    /// Subtopics, questions and answers stored for each topic.
    counts: HashMap<&'a str, [usize; 3]>,
    /// Subtopics and questions matching the filter.
    matches: Cell<usize>,
    /// The nodes to open so the matches show.
    open: RefCell<Vec<Vec<Node>>>,
}

impl<'a> Tree<'a> {
    fn new(model: &'a Model, needle: &'a str) -> Self {
        let data = &model.data;
        let mut tree = Self {
            model,
            needle,
            subtopics: HashMap::new(),
            questions: HashMap::new(),
            missing: HashMap::new(),
            counts: HashMap::new(),
            matches: Cell::new(0),
            open: RefCell::new(Vec::new()),
        };
        let ids: BTreeSet<&Id> = data.subtopics.iter().map(|s| &s.id).collect();
        for subtopic in &data.subtopics {
            tree.subtopics
                .entry(subtopic.topic.as_str())
                .or_default()
                .push(subtopic);
            tree.counts.entry(subtopic.topic.as_str()).or_default()[0] += 1;
        }
        for question in &data.questions {
            tree.counts.entry(question.topic.as_str()).or_default()[1] += 1;
            if ids.contains(&question.subtopic_id) {
                tree.questions
                    .entry(&question.subtopic_id)
                    .or_default()
                    .push(question);
            } else {
                tree.missing
                    .entry(question.topic.as_str())
                    .or_default()
                    .push(question);
            }
        }
        for example in &data.answers {
            tree.counts.entry(example.topic.as_str()).or_default()[2] += 1;
        }
        tree
    }

    /// Whether `text` matches the filter, counting each match.
    fn hit(&self, text: &str) -> bool {
        let hit = !self.needle.is_empty() && normalize(text).contains(self.needle);
        if hit {
            self.matches.set(self.matches.get() + 1);
        }
        hit
    }

    /// The node of `topic`, `None` when the filter leaves nothing of it.
    fn topic(&self, topic: &str) -> std::io::Result<Option<Item>> {
        let node = Node::Topic(topic.to_string());
        let mut children = Vec::new();
        for subtopic in self.subtopics.get(topic).into_iter().flatten() {
            let questions = self
                .questions
                .get(&subtopic.id)
                .map_or(&[][..], Vec::as_slice);
            let group = Node::Subtopic(subtopic.id.clone());
            let matched = self.hit(&subtopic.name);
            if let Some(item) =
                self.group([node.clone(), group], &subtopic.name, matched, questions)?
            {
                children.push(item);
            }
        }
        if let Some(missing) = self.missing.get(topic) {
            let group = Node::MissingSubtopic(topic.to_string());
            if let Some(item) =
                self.group([node.clone(), group], "(missing subtopic)", false, missing)?
            {
                children.push(item);
            }
        }
        if !self.needle.is_empty() {
            if children.is_empty() {
                return Ok(None);
            }
            self.open.borrow_mut().push(vec![node.clone()]);
        }
        let label = self.topic_label(topic);
        TreeItem::new(node, label, children).map(Some)
    }

    /// A topic's name, `(not configured)` after it when it is not in
    /// `overbrainer.toml`, then its counts when they fit the label's room
    /// whole.
    fn topic_label(&self, topic: &str) -> Line<'static> {
        let [subtopics, questions, answers] = self.counts.get(topic).copied().unwrap_or_default();
        let mut label = Line::from(topic.to_string());
        if !self.model.is_configured(topic) {
            label.push_span(Span::styled(" (not configured)", self.model.warn));
        }
        let counts = Span::raw(format!("  {subtopics} sub, {questions} q, {answers} a"));
        if self
            .model
            .room
            .is_none_or(|room| label.width() + counts.width() <= room)
        {
            label.push_span(counts);
        }
        label
    }

    /// The group at `path` (its topic, then the group itself: a subtopic or the
    /// missing-subtopic group); `None` when the filter leaves nothing of it. A
    /// matching group keeps every question, and a group holding a matching
    /// question is opened.
    fn group(
        &self,
        path: [Node; 2],
        name: &str,
        matched: bool,
        questions: &[&Question],
    ) -> std::io::Result<Option<Item>> {
        let mut children = Vec::new();
        let mut hits = false;
        for question in questions {
            let hit = self.hit(&question.text);
            hits |= hit;
            if self.needle.is_empty() || matched || hit {
                children.push(self.question(question)?);
            }
        }
        if children.is_empty() && !self.needle.is_empty() && !matched {
            return Ok(None);
        }
        if hits {
            self.open.borrow_mut().push(path.to_vec());
        }
        let [_, node] = path;
        let answers = questions
            .iter()
            .filter(|q| self.model.first_answer.contains_key(&q.id))
            .count();
        let label = format!("{name}  {} q, {answers} a", questions.len());
        TreeItem::new(node, Line::from(label), children).map(Some)
    }

    fn question(&self, question: &Question) -> std::io::Result<Item> {
        let text = question
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let Some(class) = self.model.class(&question.id) else {
            return Ok(TreeItem::new_leaf(
                Node::Question(question.id.clone()),
                format!("[ ] {text}"),
            ));
        };
        let (marker, note) = match class {
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
    /// The style of the `(not configured)` label of a topic.
    pub(super) warn: Style,
}

impl DatasetView {
    /// Shows a newly loaded `data`, keeping the selection and open nodes when the
    /// selected node is still shown, else selecting the first node.
    pub(super) fn loaded(&mut self, data: Dataset, topics: &[TopicInfo]) {
        let room = self.model.as_ref().and_then(|model| model.room);
        self.model = Some(Model::new(data, topics, (&self.filter, room), self.warn));
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

    /// Applies `filter`, rebuilds the tree and opens the nodes holding a match.
    pub(super) fn apply_filter(&mut self, filter: String) {
        self.filter = filter;
        if let Some(model) = &mut self.model {
            model.filter(&self.filter);
            for path in &model.open {
                self.tree.open(path.clone());
            }
        }
        self.tree.select(Vec::new());
        self.step(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::snapshots::{MOVED, dataset, path_to, topics};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

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
    fn stats_follow_split_rules_per_topic_and_overall() -> TestResult {
        let view = loaded();
        let model = view.model.as_ref().ok_or("no model")?;
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
        Ok(())
    }

    #[test]
    fn duplicate_ids_in_a_file_show_an_error_instead_of_the_tree() -> TestResult {
        let mut data = dataset();
        let copy = data.questions[0].clone();
        data.questions.push(copy);
        let mut view = DatasetView::default();
        view.loaded(data, &topics());
        let model = view.model.as_ref().ok_or("no model")?;
        assert!(
            model
                .items
                .as_ref()
                .is_err_and(|error| error.starts_with("cannot show the dataset tree")),
        );
        Ok(())
    }

    #[test]
    fn a_filter_counts_every_match_and_opens_what_holds_them() -> TestResult {
        let mut view = loaded();
        view.apply_filter("BORROW".into());
        let model = view.model.as_ref().ok_or("no model")?;
        assert_eq!(model.matches, Some(4), "the subtopic and its 3 questions");
        let shown = labels(&view);
        assert_eq!(shown.len(), 5, "{shown:#?}");
        let nll = path_to("When does NLL end a borrow?", false);
        assert!(view.shown(&nll), "{shown:#?}");
        Ok(())
    }

    #[test]
    fn the_tree_and_the_detail_use_the_first_of_duplicate_answers() -> TestResult {
        let mut data = dataset();
        let mut second = data.answers[0].clone();
        second.meta.excluded = Some(Exclusion::Refused);
        data.answers.push(second);
        let model = Model::new(data, &topics(), ("", None), Style::new());
        let id = &model.data.answers[0].id;
        assert_eq!(model.class(id), Some(SplitClass::Usable));
        let first = model.answer(id).ok_or("no answer")?;
        assert_eq!(first.meta.excluded, None, "the first, usable answer");
        Ok(())
    }

    #[test]
    fn a_model_of_thousands_of_answers_classes_each_one() -> TestResult {
        let base = dataset();
        let template = base.answers[0].clone();
        let mut data = Dataset {
            subtopics: base.subtopics.clone(),
            questions: Vec::new(),
            answers: Vec::new(),
            rejected: Vec::new(),
        };
        for n in 0..5000_usize {
            let mut question = base.questions[0].clone();
            question.text = format!("Question {n} about a borrow?");
            question.id = Id::question(&question.subtopic_id, &question.text);
            let mut example = template.clone();
            example.id = question.id.clone();
            example.meta.excluded = (n.is_multiple_of(5)).then_some(Exclusion::Empty);
            if n.is_multiple_of(7) {
                example.id = Id::question(&question.subtopic_id, &format!("gone {n}"));
            }
            data.questions.push(question);
            data.answers.push(example);
        }
        let mut view = DatasetView::default();
        view.loaded(data, &topics());
        view.apply_filter("borrow".into());
        let model = view.model.as_ref().ok_or("no model")?;
        assert_eq!(model.matches, Some(5001), "Borrowing and every question");
        for (n, example) in model.data.answers.iter().enumerate() {
            let expected = if n.is_multiple_of(7) {
                SplitClass::Orphaned
            } else if n.is_multiple_of(5) {
                SplitClass::Excluded(Exclusion::Empty)
            } else {
                SplitClass::Usable
            };
            let class = model.classes.get(n).copied();
            assert_eq!(class, Some(expected), "answer {n}");
            if !n.is_multiple_of(7) {
                assert_eq!(model.class(&example.id), Some(expected), "answer {n}");
            }
        }
        let all = model.stats.get(&None).ok_or("no stats")?;
        assert_eq!((all.answers, all.usable, all.orphaned), (5000, 3428, 715));
        Ok(())
    }
}
