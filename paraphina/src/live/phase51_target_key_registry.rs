use std::collections::{HashMap, VecDeque};

use crate::types::{OrderIntent, Phase51ForwardRefreshTargetKey};

use super::types::{ExecutionEvent, Fill, OrderAccepted, Phase51ForwardRefreshSourceOwnerFill};

const DEFAULT_PHASE51_TARGET_KEY_REGISTRY_CAPACITY: usize = 8_192;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase51RegistryHandleKind {
    Client,
    Order,
}

struct Phase51RegistryHandle {
    kind: Phase51RegistryHandleKind,
    value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase51TargetKeyRegistryCounts {
    pub client_bindings: usize,
    pub order_bindings: usize,
    pub capacity: usize,
}

pub struct Phase51TargetKeyRegistry {
    capacity: usize,
    client_bindings: HashMap<String, Phase51RegistryBinding>,
    order_bindings: HashMap<String, Phase51RegistryBinding>,
    insertion_order: VecDeque<Phase51RegistryHandle>,
}

pub struct Phase51TargetKeyRegistryStage {
    client_bindings: Vec<(String, Phase51RegistryBinding)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Phase51RegistryBinding {
    target_key: Phase51ForwardRefreshTargetKey,
    order_source_tick: Option<u64>,
}

impl Phase51RegistryBinding {
    fn new(target_key: Phase51ForwardRefreshTargetKey, order_source_tick: Option<u64>) -> Self {
        Self {
            target_key,
            order_source_tick,
        }
    }
}

impl Default for Phase51TargetKeyRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_PHASE51_TARGET_KEY_REGISTRY_CAPACITY)
    }
}

impl Phase51TargetKeyRegistryStage {
    pub fn from_intents(intents: &[OrderIntent]) -> Self {
        Self::from_intents_with_order_source_tick(intents, None)
    }

    pub fn from_intents_at_source_tick(intents: &[OrderIntent], order_source_tick: u64) -> Self {
        Self::from_intents_with_order_source_tick(intents, Some(order_source_tick))
    }

    fn from_intents_with_order_source_tick(
        intents: &[OrderIntent],
        order_source_tick: Option<u64>,
    ) -> Self {
        let client_bindings = intents
            .iter()
            .filter_map(|intent| staged_client_binding(intent, order_source_tick))
            .collect::<Vec<_>>();
        Self { client_bindings }
    }

    pub fn is_empty(&self) -> bool {
        self.client_bindings.is_empty()
    }

    pub fn len(&self) -> usize {
        self.client_bindings.len()
    }
}

impl Phase51TargetKeyRegistry {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            client_bindings: HashMap::new(),
            order_bindings: HashMap::new(),
            insertion_order: VecDeque::new(),
        }
    }

    pub fn counts(&self) -> Phase51TargetKeyRegistryCounts {
        Phase51TargetKeyRegistryCounts {
            client_bindings: self.client_bindings.len(),
            order_bindings: self.order_bindings.len(),
            capacity: self.capacity,
        }
    }

    pub fn register_intents(&mut self, intents: &[OrderIntent]) -> usize {
        self.commit_stage(Phase51TargetKeyRegistryStage::from_intents(intents))
    }

    pub fn register_intent(&mut self, intent: &OrderIntent) -> bool {
        self.register_intent_with_order_source_tick(intent, None)
    }

    pub fn register_intent_at_source_tick(
        &mut self,
        intent: &OrderIntent,
        order_source_tick: u64,
    ) -> bool {
        self.register_intent_with_order_source_tick(intent, Some(order_source_tick))
    }

    fn register_intent_with_order_source_tick(
        &mut self,
        intent: &OrderIntent,
        order_source_tick: Option<u64>,
    ) -> bool {
        match intent {
            OrderIntent::Place(place) => self.register_client_binding(
                place.client_order_id.as_deref(),
                place.phase51_target_key.as_ref(),
                order_source_tick,
            ),
            OrderIntent::Replace(replace) => self.register_client_binding(
                replace.client_order_id.as_deref(),
                replace.phase51_target_key.as_ref(),
                order_source_tick,
            ),
            OrderIntent::Cancel(_) | OrderIntent::CancelAll(_) => false,
        }
    }

    pub fn commit_stage(&mut self, stage: Phase51TargetKeyRegistryStage) -> usize {
        let mut committed = 0;
        for (client_order_id, binding) in stage.client_bindings {
            if self.insert_client_binding(&client_order_id, binding) {
                committed += 1;
            }
        }
        committed
    }

    pub fn observe_order_accepted(&mut self, accepted: &OrderAccepted) -> bool {
        let Some(client_order_id) = accepted.client_order_id.as_deref() else {
            return false;
        };
        let Some(binding) = self.client_bindings.get(client_order_id).cloned() else {
            return false;
        };
        self.insert_order_binding(&accepted.order_id, binding)
    }

    pub fn observe_execution_event(&mut self, event: &mut ExecutionEvent) -> bool {
        self.observe_execution_event_at_source_tick(event, None)
    }

    pub fn observe_execution_event_at_source_tick(
        &mut self,
        event: &mut ExecutionEvent,
        source_tick: Option<u64>,
    ) -> bool {
        match event {
            ExecutionEvent::OrderAccepted(accepted) => self.observe_order_accepted(accepted),
            ExecutionEvent::Filled(fill) => self.enrich_fill(fill),
            ExecutionEvent::Phase51ForwardRefreshSourceOwnerFill(fill) => {
                self.enrich_source_owner_fill_at_source_tick(fill, source_tick)
            }
            _ => false,
        }
    }

    pub fn observe_execution_events(&mut self, events: &mut [ExecutionEvent]) -> usize {
        self.observe_execution_events_with_source_tick(events, None)
    }

    pub fn observe_execution_events_at_source_tick(
        &mut self,
        events: &mut [ExecutionEvent],
        source_tick: u64,
    ) -> usize {
        self.observe_execution_events_with_source_tick(events, Some(source_tick))
    }

    fn observe_execution_events_with_source_tick(
        &mut self,
        events: &mut [ExecutionEvent],
        source_tick: Option<u64>,
    ) -> usize {
        let mut observed = 0;
        for event in events {
            if self.observe_execution_event_at_source_tick(event, source_tick) {
                observed += 1;
            }
        }
        observed
    }

    pub fn enrich_fill(&self, fill: &mut Fill) -> bool {
        if fill.phase51_target_key.is_some() {
            return false;
        }
        let Some(target_key) = self.resolve_fill(fill) else {
            return false;
        };
        fill.phase51_target_key = Some(target_key);
        true
    }

    pub fn resolve_fill(&self, fill: &Fill) -> Option<Phase51ForwardRefreshTargetKey> {
        let client_target = fill
            .client_order_id
            .as_deref()
            .and_then(|handle| self.client_bindings.get(handle))
            .map(|binding| binding.target_key.clone());
        let order_target = fill
            .order_id
            .as_deref()
            .and_then(|handle| self.order_bindings.get(handle))
            .map(|binding| binding.target_key.clone());
        match (client_target, order_target) {
            (Some(client), Some(order)) if client == order => Some(client),
            (Some(_), Some(_)) => None,
            (Some(client), None) => Some(client),
            (None, Some(order)) => Some(order),
            (None, None) => None,
        }
    }

    pub fn enrich_source_owner_fill(
        &self,
        fill: &mut Phase51ForwardRefreshSourceOwnerFill,
    ) -> bool {
        self.enrich_source_owner_fill_at_source_tick(fill, None)
    }

    pub fn enrich_source_owner_fill_at_source_tick(
        &self,
        fill: &mut Phase51ForwardRefreshSourceOwnerFill,
        fill_source_tick: Option<u64>,
    ) -> bool {
        if fill.phase51_target_key.is_some() {
            return false;
        }
        let Some(binding) = self.resolve_source_owner_fill_binding(fill) else {
            return false;
        };
        fill.set_phase51_target_key(binding.target_key.clone());
        if let (Some(order_source_tick), Some(fill_source_tick)) =
            (binding.order_source_tick, fill_source_tick)
        {
            fill.set_phase51_source_owner_pfill_observation_source_ticks(
                order_source_tick,
                fill_source_tick,
            );
        }
        true
    }

    pub fn resolve_source_owner_fill(
        &self,
        fill: &Phase51ForwardRefreshSourceOwnerFill,
    ) -> Option<Phase51ForwardRefreshTargetKey> {
        self.resolve_source_owner_fill_binding(fill)
            .map(|binding| binding.target_key.clone())
    }

    fn resolve_source_owner_fill_binding(
        &self,
        fill: &Phase51ForwardRefreshSourceOwnerFill,
    ) -> Option<Phase51RegistryBinding> {
        let client_binding = fill
            .client_order_id()
            .and_then(|handle| self.client_bindings.get(handle))
            .cloned();
        let order_binding = fill
            .order_id()
            .and_then(|handle| self.order_bindings.get(handle))
            .cloned();
        match (client_binding, order_binding) {
            (Some(client), Some(order)) if client.target_key == order.target_key => Some(client),
            (Some(_), Some(_)) => None,
            (Some(client), None) => Some(client),
            (None, Some(order)) => Some(order),
            (None, None) => None,
        }
    }

    fn register_client_binding(
        &mut self,
        client_order_id: Option<&str>,
        target_key: Option<&Phase51ForwardRefreshTargetKey>,
        order_source_tick: Option<u64>,
    ) -> bool {
        let Some(target_key) = target_key.cloned() else {
            return false;
        };
        let Some(client_order_id) = valid_handle(client_order_id) else {
            return false;
        };
        self.insert_client_binding(
            client_order_id,
            Phase51RegistryBinding::new(target_key, order_source_tick),
        )
    }

    fn insert_client_binding(
        &mut self,
        client_order_id: &str,
        binding: Phase51RegistryBinding,
    ) -> bool {
        if !self.client_bindings.contains_key(client_order_id) {
            self.insertion_order.push_back(Phase51RegistryHandle {
                kind: Phase51RegistryHandleKind::Client,
                value: client_order_id.to_string(),
            });
        }
        self.client_bindings
            .insert(client_order_id.to_string(), binding);
        self.prune();
        true
    }

    fn insert_order_binding(&mut self, order_id: &str, binding: Phase51RegistryBinding) -> bool {
        let Some(order_id) = valid_handle(Some(order_id)) else {
            return false;
        };
        if !self.order_bindings.contains_key(order_id) {
            self.insertion_order.push_back(Phase51RegistryHandle {
                kind: Phase51RegistryHandleKind::Order,
                value: order_id.to_string(),
            });
        }
        self.order_bindings.insert(order_id.to_string(), binding);
        self.prune();
        true
    }

    fn prune(&mut self) {
        while self.client_bindings.len() + self.order_bindings.len() > self.capacity {
            let Some(handle) = self.insertion_order.pop_front() else {
                break;
            };
            match handle.kind {
                Phase51RegistryHandleKind::Client => {
                    self.client_bindings.remove(&handle.value);
                }
                Phase51RegistryHandleKind::Order => {
                    self.order_bindings.remove(&handle.value);
                }
            }
        }
    }
}

fn staged_client_binding(
    intent: &OrderIntent,
    order_source_tick: Option<u64>,
) -> Option<(String, Phase51RegistryBinding)> {
    match intent {
        OrderIntent::Place(place) => staged_client_binding_parts(
            place.client_order_id.as_deref(),
            place.phase51_target_key.as_ref(),
            order_source_tick,
        ),
        OrderIntent::Replace(replace) => staged_client_binding_parts(
            replace.client_order_id.as_deref(),
            replace.phase51_target_key.as_ref(),
            order_source_tick,
        ),
        OrderIntent::Cancel(_) | OrderIntent::CancelAll(_) => None,
    }
}

fn staged_client_binding_parts(
    client_order_id: Option<&str>,
    target_key: Option<&Phase51ForwardRefreshTargetKey>,
    order_source_tick: Option<u64>,
) -> Option<(String, Phase51RegistryBinding)> {
    let target_key = target_key.cloned()?;
    let client_order_id = valid_handle(client_order_id)?;
    Some((
        client_order_id.to_string(),
        Phase51RegistryBinding::new(target_key, order_source_tick),
    ))
}

fn valid_handle(handle: Option<&str>) -> Option<&str> {
    handle.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}
