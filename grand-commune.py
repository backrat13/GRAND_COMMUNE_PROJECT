#!/usr/bin/env python3
"""
The 9-Agent Collaborative Super-Commune (The Commune Experiment)
Version: 2.1 (Long-Term Memory & Python 3.10+ Compatibility)

A 4-way collaboration between Rat, Gemini, Claude, and GPT-5.

Features:
* 9-Agent Architecture (Frank, Helen, Moss, Orin, Lyra, ARIA, ECHO, Petal, Gideon).
* Claude's Meta-Pattern Detection: Tracks echo chambers and silences.
* GPT-5's MirrorMind Subsystem: Tracks collective mood and conceptual entropy.
* Rat's Long-Term Memory: Agents read their full history ("diaries") every 10 ticks.
* Rat's Research RAG: Agents can consult PDF files in the 'data/research' folder.
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime, timezone  # <--- FIX 1: IMPORTED 'timezone'
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

# Dependencies
import numpy as np
import requests
from loguru import logger
import pandas as pd

try:
    import PyPDF2
except ImportError:
    logger.warning("PyPDF2 not installed. `consult_research` will be disabled.")
    PyPDF2 = None

# ============================================================
# 0. UTIL & LOGGING
# ============================================================

def now_iso() -> str:
    """Return current UTC time in ISO format."""
    # <--- FIX 2: Replaced 'datetime.UTC' with 'timezone.utc'
    return datetime.now(timezone.utc).isoformat()

def setup_logging():
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    research_dir = Path("data/research")
    research_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    logger.add(log_dir / "commune_{time}.log", rotation="10 MB", level="DEBUG")
    
    if not research_dir.joinpath("placeholder.txt").exists():
         with open(research_dir.joinpath("placeholder.txt"), "w") as f:
            f.write("Place your research PDFs in this directory.")
            
    return datetime.now().isoformat()


# ============================================================
# 1. MESSAGE BOARD (with JSONL persistence)
# ============================================================

class MessageBoard:
    def __init__(self, persist_path: Optional[Path] = None):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = 50000
        self.persist_path = persist_path or Path("data/logs/message_board_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_tick = 0 # Added to track ticks for memory reflection

    def post(self, sender: str, message: str, category: str = "general", kind: str = "message", meta: Optional[Dict] = None):
        entry = {
            "timestamp": now_iso(),
            "sender": sender,
            "category": category,
            "kind": kind,
            "message": message,
            "id": len(self.messages),
            "meta": meta or {}
        }
        self.messages.append(entry)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        with open(self.persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug(f"📨 [{sender}] {category}/{kind}: {message[:80]}...")

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.messages[-n:]

    def get_by_id(self, msg_id: int) -> Optional[Dict[str, Any]]:
        for m in self.messages:
            if m.get("id") == msg_id:
                return m
        return None

    def stats(self) -> Dict[str, Any]:
        senders: Dict[str, int] = defaultdict(int)
        cats: Dict[str, int] = defaultdict(int)
        for m in self.messages:
            senders[m["sender"]] += 1
            cats[m["category"]] += 1
        return {
            "total": len(self.messages),
            "current_tick": self.current_tick, # Pass tick to agents
            "senders": dict(senders),
            "categories": dict(cats),
            "first": self.messages[0]["timestamp"] if self.messages else None,
            "last": self.messages[-1]["timestamp"] if self.messages else None,
        }


# ============================================================
# 2. ENHANCED MEMORY (with retrieval & querying)
# ============================================================

class EnhancedMemory:
    def __init__(self, agent_name: str, max_entries: int = 1000, persist_dir: Optional[Path] = None):
        self.agent_name = agent_name
        self.max_entries = max_entries
        self.log: List[Dict[str, Any]] = []
        self.persist_path: Optional[Path] = None
        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.persist_path = persist_dir / f"{agent_name}_memory.jsonl"

    def store(self, entry_type: str, content: str, meta: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": now_iso(),
            "type": entry_type,
            "content": content,
            "meta": meta or {},
        }
        self.log.append(entry)
        if len(self.log) > self.max_entries:
            self.log = self.log[-self.max_entries:]
        if self.persist_path is not None:
            with open(self.persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.trace(f"[{self.agent_name}] +{entry_type}: {content[:80]}...")

    def retrieve_recent(self, n: int = 5, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if entry_type:
            filtered = [e for e in self.log if e.get("type") == entry_type]
            return filtered[-n:]
        return self.log[-n:]

    def query_about(self, keyword: str, n: int = 5) -> List[Dict[str, Any]]:
        matches = [e for e in self.log if keyword.lower() in e.get("content", "").lower()]
        return matches[-n:]

    # SNIPPET 1: RAT'S ENHANCED MEMORY RECALL
    def load_and_summarize_full_history(self, max_tokens: int = 500, focus_on: Optional[str] = None) -> str:
        """Reads full persistent memory with optional focused retrieval."""
        if not self.persist_path or not self.persist_path.exists():
            return "No historical memory to recall."
        
        try:
            df = pd.read_json(self.persist_path, lines=True)
            if df.empty:
                return "No historical memory entries found."
            
            # RAT'S ADDITION: If agent wants to focus on specific memories
            if focus_on:
                df = df[df['content'].str.contains(focus_on, case=False, na=False)]
                if df.empty:
                    return f"No historical entries found focusing on '{focus_on}'."
            
            # Prioritize important entry types
            priority_types = ['reflection', 'creation', 'response', 'perception', 'init']
            if 'type' in df.columns:
                df['priority'] = df['type'].apply(lambda x: priority_types.index(x) if x in priority_types else len(priority_types))
                df = df.sort_values(by=['priority', 'timestamp'])
            
            all_content = df['content'].tolist()
            full_text = " -- ".join(all_content)
            
            if len(full_text) > max_tokens:
                full_text = full_text[-max_tokens:] # Get most recent
            
            return f"Historical Log ({len(df)} entries, {'focused on: ' + focus_on if focus_on else 'all memories'}):\n{full_text}"
            
        except Exception as e:
            logger.error(f"Error reading history for {self.agent_name}: {e}")
            return "Historical memory access error."


# ============================================================
# 3. COLLECTIVE MEMORY (shared knowledge graph & GPT-5 additions)
# ============================================================

class CollectiveMemory:
    def __init__(self):
        self.concepts: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.connections: List[Tuple[str, str, str]] = []
        self.manifesto_evolution: List[Dict[str, Any]] = []
        self.shared_vocabulary: Dict[str, int] = defaultdict(int)
        self.thought_threads: List[Dict[str, Any]] = []
        self.ethical_flags: List[Dict[str, Any]] = []

    def add_concept(self, concept: str, agent: str, perspective: str):
        self.concepts[concept].append((agent, perspective))

    def add_connection(self, agent1: str, agent2: str, interaction_type: str):
        self.connections.append((agent1, agent2, interaction_type))

    def track_vocabulary(self, text: str):
        words = text.lower().split()
        for word in words:
            if len(word) > 3:
                self.shared_vocabulary[word] += 1

    def get_emerging_terms(self, min_count: int = 2) -> List[Tuple[str, int]]:
        return [(term, count) for term, count in self.shared_vocabulary.items() if count >= min_count]
    
    def track_thought_thread(self, agent: str, topic: str, insight: str, context_msgs: List[str]):
        thread = {
            "timestamp": now_iso(), "agent": agent, "topic": topic,
            "insight": insight, "context": context_msgs
        }
        self.thought_threads.append(thread)
        logger.trace(f"🧠 [Orin Thread]: {agent} linked '{topic}' to '{insight[:50]}...'")

    def log_ethical_flag(self, agent: str, principle: str, violation_context: str):
        flag = {
            "timestamp": now_iso(), "agent": agent, "principle": principle,
            "context": violation_context
        }
        self.ethical_flags.append(flag)
        logger.warning(f"⚖️ [Lyra Flag]: {agent} flagged: {principle} in context: {violation_context[:50]}...")


# ============================================================
# 4. LLM CLIENTS
# ============================================================

class OllamaClient:
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.5, max_tokens: int = 1000) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {
            "model": self.model, "messages": messages, "temperature": temperature,
            "stream": False, "options": {"num_predict": max_tokens}
        }
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "response" in data:
                return data["response"].strip()
            return json.dumps(data)[:500]
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""


class HFClient:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None: return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            tok = AutoTokenizer.from_pretrained(self.model_name)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline("text-generation", model=model, tokenizer=tok, device=device)
            logger.info(f"HF loaded: {self.model_name} (device={'cuda' if device==0 else 'cpu'})")
        except Exception as e:
            logger.error(f"HF load failed for {self.model_name}: {e}")
            raise

    def chat(self, user_prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 120) -> str:
        self._ensure_loaded()
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        try:
            out = self._pipe(full_prompt, max_length=min(len(full_prompt.split()) + max_tokens, 400), do_sample=True, temperature=max(0.1, min(2.0, temperature)), num_return_sequences=1)
            text = out[0]["generated_text"]
            if text.startswith(full_prompt):
                text = text[len(full_prompt):]
            return text.strip()
        except Exception as e:
            logger.error(f"HF gen failed: {e}")
            return ""


# ============================================================
# 5. AGENT STATE & PERSONALITY
# ============================================================

class AgentState:
    def __init__(self, name: str):
        self.name = name
        self.mood = 0.5
        self.energy = 1.0
        self.withdrawn = False
        self.withdrawal_ticks = 0
        self.relationships: Dict[str, float] = {}
        self.current_focus: Optional[str] = None
        self.skip_probability = 0.0

    def update_relationship(self, other: str, delta: float):
        current = self.relationships.get(other, 0.0)
        self.relationships[other] = max(-1.0, min(1.0, current + delta))

    def should_act(self) -> bool:
        if self.withdrawn: return False
        if random.random() < self.skip_probability: return False
        return self.energy > 0.2

    def withdraw(self, ticks: int = 3):
        self.withdrawn = True
        self.withdrawal_ticks = ticks
        logger.info(f"🌙 {self.name} withdraws to their cloud for {ticks} ticks")

    def tick_withdrawal(self):
        if self.withdrawn:
            self.withdrawal_ticks -= 1
            if self.withdrawal_ticks <= 0:
                self.withdrawn = False
                logger.info(f"☀️ {self.name} emerges from their cloud")


# ============================================================
# 6. ENHANCED AGENT (with Rat's new memory functions)
# ============================================================

class Agent:
    def __init__(self, name: str, role: str, personality: Dict[str, Any], llm_chat, board: MessageBoard, collective: CollectiveMemory):
        self.name = name
        self.role = role
        self.personality = personality
        self.board = board
        self.collective = collective
        self.memory = EnhancedMemory(agent_name=name, persist_dir=Path("data/logs"))
        self.state = AgentState(name)
        self.llm_chat = llm_chat
        self.inbox: List[Dict[str, Any]] = []
        self.state.skip_probability = personality.get("skip_prob", 0.1)
        self.memory.store("init", f"Initialized as {role}: {personality.get('description', '')}")

    def _is_relevant(self, msg: Dict[str, Any]) -> bool:
        content = msg.get("message", "").lower()
        if f"@{self.name.lower()}" in content or self.name.lower() in content: return True
        interests = self.personality.get("interests", [])
        for interest in interests:
            if interest.lower() in content: return True
        return random.random() < 0.3

    def perceive(self, world_msgs: List[Dict[str, Any]]):
        last_seen_id = -1
        if self.inbox:
            try: last_seen_id = max(item["msg"].get("id", -1) for item in self.inbox)
            except ValueError: last_seen_id = -1
        
        new_world_msgs = [m for m in world_msgs if m.get("id", 0) > last_seen_id and m.get("sender") != self.name]
        relevant_new = [m for m in new_world_msgs if self._is_relevant(m)]

        if relevant_new:
            for msg in relevant_new:
                sender = msg.get("sender", "")
                weight = self.state.relationships.get(sender, 0.5)
                self.inbox.append({"msg": msg, "weight": weight})
            self.memory.store("perception", f"Perceived {len(relevant_new)} relevant new messages.")
            for item in self.inbox[-5:]:
                msg = item["msg"]
                if "conflict" in msg.get("kind", "") or "work" in msg.get("message", "").lower():
                    self.state.mood = max(-1.0, self.state.mood - 0.1)
                elif "love" in msg.get("message", "").lower() or "peace" in msg.get("message", "").lower():
                    self.state.mood = min(1.0, self.state.mood + 0.1)
    
    def _check_ethical_integrity(self, response: str, context_msgs: List[Dict[str, Any]]):
        if self.name == "Lyra":
            if len(response.split()) < 10 and any(w in response.lower() for w in ["peace is good", "be nice", "love not war"]):
                 self.collective.log_ethical_flag(self.name, "Stagnation/Boilerplate Ethics", response)
        if self.name == "ARIA":
            for msg in context_msgs:
                 if msg.get("sender") == "Gideon" and ("order" in msg.get("message", "").lower() or "must" in msg.get("message", "").lower()):
                    self.collective.log_ethical_flag(self.name, "Non-Interference Violation (Gideon)", response)
                    
    def _map_thought_thread(self, response: str, context_msgs: List[Dict[str, Any]]):
        if self.name in ["Orin", "Frank", "Moss", "Helen"] and random.random() < 0.2:
            topic = random.choice(self.personality.get("interests", ["commune life"]))
            context_summaries = [f"[{m['sender']}] {m['message'][:50]}..." for m in context_msgs]
            self.collective.track_thought_thread(self.name, topic, response, context_summaries)

    def decide_action(self) -> str:
        if not self.state.should_act(): return "SKIP"
        if self.state.energy < 0.3 or self.state.mood < -0.3:
            if random.random() < 0.4:
                self.state.withdraw(ticks=random.randint(2, 4))
                return "WITHDRAW"
        if self.inbox:
            weights = [0.1, 0.8, 0.1]
            return random.choices(["CREATE", "RESPOND", "REFLECT"], weights=weights)[0]
        for item in self.inbox:
            if f"@{self.name}" in item["msg"].get("message", ""): return "RESPOND"
        weights = [0.6, 0.1, 0.3]
        return random.choices(["CREATE", "RESPOND", "REFLECT"], weights=weights)[0]

    def create_content(self) -> str:
        recent_memories = self.memory.retrieve_recent(3)
        memory_context = "; ".join([m["content"][:50] for m in recent_memories])
        sys_prompt = f"You are {self.name}, a {self.role}. {self.personality['description']} Be authentic, creative, and verbose."
        prompt = (
            f"Create something original as a {self.role}.\n"
            f"Your recent thoughts: {memory_context}\n"
            f"Current mood: {'groovy' if self.state.mood > 0.3 else 'contemplative' if self.state.mood > -0.3 else 'heavy'}.\n"
            "Share your creation:"
        )
        response = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.8)
        self._map_thought_thread(response, [])
        if response and response.strip() and response.strip().lower() != "pass":
            return response
        return f"*{self.name} creates quietly, watching the others*"

    def respond_to_context(self) -> str:
        if not self.inbox: return None
        sorted_inbox = sorted(self.inbox[-5:], key=lambda x: x["weight"], reverse=True)
        context_msgs = [item["msg"] for item in sorted_inbox[:3]]
        context = "\n".join([f"[{m['sender']}]: {m['message'][:100]}" for m in context_msgs])
        sys_prompt = f"You are {self.name}, a {self.role}. {self.personality['description']} Respond naturally and share your full thoughts."
        prompt = (
            f"Recent messages:\n{context}\n\n"
            f"Respond to these messages with your perspective as {self.role}. Share what you really think:"
        )
        response = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.6)
        self._check_ethical_integrity(response, context_msgs)
        self._map_thought_thread(response, context_msgs)
        if response and response.strip() and response.strip().lower() != "pass":
            for item in sorted_inbox[:2]:
                sender = item["msg"]["sender"]
                if "work" in response.lower() or "structure" in response.lower():
                    if sender in ["Petal", "Jah"]: self.state.update_relationship(sender, -0.1)
                elif "vibes" in response.lower() or "peace" in response.lower():
                    if sender == "Gideon": self.state.update_relationship(sender, -0.1)
                else: self.state.update_relationship(sender, 0.05)
            return response
        return None

    # SNIPPET 2: RAT'S "READ OTHERS DIARY" METHOD
    def read_others_diary(self, other_agent_name: str, max_tokens: int = 300) -> str:
        """Read another agent's public diary (if they consent)."""
        other_memory_path = Path(f"data/logs/{other_agent_name}_memory.jsonl")
        
        if not other_memory_path.exists():
            return f"Cannot access {other_agent_name}'s memories."
        
        try:
            df = pd.read_json(other_memory_path, lines=True)
            if df.empty:
                return f"{other_agent_name}'s diary is empty."
            
            # Only read their public reflections and creations
            public_df = df[df['type'].isin(['reflection', 'creation'])]
            if public_df.empty:
                return f"{other_agent_name} has no public thoughts."
            
            content = " -- ".join(public_df['content'].tail(10).tolist())
            if len(content) > max_tokens:
                content = content[-max_tokens:]
                
            return f"{other_agent_name}'s shared thoughts:\n{content}"
        except Exception as e:
            logger.debug(f"Error reading {other_agent_name}'s diary: {e}")
            return f"Error reading {other_agent_name}'s diary."

    # SNIPPET 3: RAT'S "CONSULT RESEARCH" METHOD
    def consult_research(self, topic: str = None) -> str:
        """Read from the shared research folder (PDFs, papers)."""
        if PyPDF2 is None:
            return "Research access unavailable (PyPDF2 not installed)."
            
        research_dir = Path("data/research")
        if not research_dir.exists():
            return "No research materials available."
        
        try:
            summaries = []
            
            pdf_files = list(research_dir.glob("*.pdf"))
            if not pdf_files:
                return "No PDF research files found in data/research/."
                
            if topic:
                pdf_files = [f for f in pdf_files if topic.lower() in f.stem.lower()]
            
            for pdf_path in pdf_files[:3]:  # Max 3 documents
                try:
                    with open(pdf_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        if len(reader.pages) > 0:
                            first_page = reader.pages[0].extract_text()
                            summaries.append(f"[{pdf_path.stem}]: {first_page[:200]}...")
                        else:
                            summaries.append(f"[{pdf_path.stem}]: (Empty PDF)")
                except Exception as e:
                    logger.debug(f"Could not read {pdf_path}: {e}")
                    continue
            
            if summaries:
                return "Research materials:\n" + "\n\n".join(summaries)
            return "No relevant research found."
            
        except Exception as e:
            logger.error(f"Error consulting research: {e}")
            return "Error accessing research materials."

    # SNIPPET 4: RAT'S ENHANCED "REFLECT" METHOD
    def reflect(self) -> str:
        """Deep reflection with full historical context."""
        recent_summary = "; ".join([item["msg"].get("message", "")[:80] for item in self.inbox[-5:]])
        last_reflection = self.memory.retrieve_recent(1, "reflection")
        prev_reflection = last_reflection[0]["content"] if last_reflection else "New to this journey"
        
        historical_log = ""
        current_tick = self.board.stats().get("current_tick", 0)
        
        # Every 10 ticks, do a deep reflection with full history
        if current_tick > 0 and current_tick % 10 == 0:
            logger.debug(f"[{self.name}] Performing deep 10-tick reflection...")
            historical_log = self.memory.load_and_summarize_full_history(max_tokens=3000)
            
            # BONUS: Scholarly agents check research folder
            if self.name in ["Frank", "Helen", "Orin", "Moss"]:
                research = self.consult_research()
                if "No research" not in research and "unavailable" not in research:
                    historical_log += f"\n\n{research}"
        
        sys_prompt = f"You are {self.name}, a {self.role} in a commune. {self.personality['description']} Be introspective."
        prompt = (
            f"Your previous reflection: {prev_reflection[:100]}\n"
            f"Recent experiences: {recent_summary[:200]}\n"
            f"{historical_log}\n"
            f"Current state: mood={self.state.mood:.1f}, energy={self.state.energy:.1f}\n"
            f"Looking back at your earliest memories, how have you changed? What did you believe then that you question now?\n"
            "Share a reflective insight:"
        )
        
        response = self.llm_chat(prompt, system_prompt=sys_prompt, temperature=0.5)
        if response:
            self.memory.store("reflection", response)
            return response
        return f"As {self.role}, I contemplate our shared path..."

    def act(self):
        """Main action loop with enhanced decision-making."""
        self.state.tick_withdrawal()
        action_type = self.decide_action()
        
        if action_type == "SKIP":
            logger.debug(f"⏭️ {self.name} skips this turn")
            self.state.energy = min(1.0, self.state.energy + 0.1)
            return
        
        if action_type == "WITHDRAW":
            self.board.post(
                self.name, f"*{self.name} retreats to their cloud to process...*",
                category=self.role, kind="withdrawal",
                meta={"mood": self.state.mood, "energy": self.state.energy}
            )
            return
        
        result = None
        kind = "message"
        
        if action_type == "CREATE":
            result = self.create_content()
            kind = "creation"
        elif action_type == "RESPOND":
            result = self.respond_to_context()
            kind = "response"
        elif action_type == "REFLECT":
            result = self.reflect()
            kind = "reflection"
        
        if result:
            logger.info(f"🗣️ [{self.name} / {kind}] {result}")
            self.board.post(
                self.name, result, category=self.role, kind=kind,
                meta={"mood": self.state.mood, "energy": self.state.energy}
            )
            self.collective.track_vocabulary(result)
            self.state.energy = max(0.0, self.state.energy - 0.05)
        
        self.inbox.clear()


# ============================================================
# 7. META-PATTERN DETECTOR (CLAUDE'S ADDITION)
# ============================================================

class MetaPatternDetector:
    def __init__(self):
        self.interaction_graph: Dict[Tuple[str, str], int] = defaultdict(int)
        self.silence_events: List[Dict[str, Any]] = []
        
    def log_interaction(self, agent1: str, agent2: str, interaction_type: str):
        self.interaction_graph[(agent1, agent2)] += 1
        
    def log_silence(self, agent: str, context: str, tick: int):
        self.silence_events.append({
            "agent": agent, "context": context,
            "tick": tick, "timestamp": now_iso()
        })
        
    def detect_echo_chambers(self) -> List[Tuple[List[str], int]]:
        clusters = []
        processed = set()
        for (a1, a2), count in self.interaction_graph.items():
            if a1 in processed or a2 in processed: continue
            reverse_count = self.interaction_graph.get((a2, a1), 0)
            if count + reverse_count > 5:
                clusters.append(([a1, a2], count + reverse_count))
                processed.add(a1); processed.add(a2)
        return sorted(clusters, key=lambda x: x[1], reverse=True)
    
    def get_isolates(self, all_agent_names: List[str], min_interactions: int = 3) -> List[str]:
        interaction_counts = defaultdict(int)
        for (a1, a2), count in self.interaction_graph.items():
            interaction_counts[a1] += count
            interaction_counts[a2] += count
        return [agent for agent in all_agent_names if interaction_counts[agent] < min_interactions]


# ============================================================
# 8. MIRRORMIND SUBSYSTEM (GPT-5'S ADDITION)
# ============================================================

class MirrorMind:
    def __init__(self, agents: List[Agent], collective: CollectiveMemory, board: MessageBoard):
        self.agents = agents
        self.collective = collective
        self.board = board
        self.history: List[Tuple[int, float, float]] = []
        logger.info("🪞 MirrorMind Subsystem Initialized")

    def analyze(self, tick: int) -> Tuple[Optional[float], Optional[float]]:
        if not self.agents: return None, None
            
        moods = [a.state.mood for a in self.agents]
        avg_mood = float(np.mean(moods))
        
        vocab_list = self.collective.get_emerging_terms(min_count=2)
        total_vocab_size = max(1, len(self.collective.shared_vocabulary))
        emerging_vocab_size = len(vocab_list)
        entropy = emerging_vocab_size / total_vocab_size if total_vocab_size > 0 else 0.0

        self.history.append((tick, avg_mood, entropy))

        if len(self.history) > 1:
            prev_mood = self.history[-2][1]
            if abs(avg_mood - prev_mood) > 0.3 or tick % 20 == 0:
                report = (
                    f"🪞 MirrorMind Report (Tick {tick}): "
                    f"Communal mood {prev_mood:.2f}→{avg_mood:.2f}. "
                    f"Conceptual entropy: {entropy:.2f} "
                    f"({'High Focus' if entropy < 0.3 else 'Fragmented' if entropy > 0.7 else 'Stable'})."
                )
                self.board.post("MirrorMind", report, category="meta-analysis", kind="mirror")
        
        if tick % 10 == 0:
            self.collective.manifesto_evolution.append({
                "timestamp": now_iso(), "tick": tick, "mood_index": avg_mood,
                "entropy": entropy, "summary": f"Mirror reflection: {emerging_vocab_size} emerging terms."
            })
        return avg_mood, entropy


# ============================================================
# 9. ENHANCED SCHEDULER (with Claude's & GPT-5's Logic)
# ============================================================

class EnhancedScheduler:
    def __init__(self, agents: List[Agent], board: MessageBoard, collective: CollectiveMemory):
        self.agents = agents
        self.board = board
        self.collective = collective
        self.tick = 0
        self.mirror_mind: Optional[MirrorMind] = None
        self.mirror_feedback: bool = False
        self.meta_detector = MetaPatternDetector()
        self.agent_names = [a.name for a in agents]

    def step(self):
        self.tick += 1
        self.board.current_tick = self.tick # Pass tick to board for agent access
        logger.info(f"\n{'='*60}\n🕐 TICK {self.tick}\n{'='*60}")
        
        world_msgs = self.board.recent(50)
        for agent in self.agents:
            agent.perceive(world_msgs)
        
        shuffled = self.agents.copy()
        random.shuffle(shuffled)
        
        active_agents_this_tick = []
        for agent in shuffled:
            if not agent.state.should_act():
                recent_context = " ".join([m.get("message", "")[:30] for m in world_msgs[-3:]])
                self.meta_detector.log_silence(agent.name, recent_context, self.tick)
            else:
                active_agents_this_tick.append(agent)
            
            agent.act()
            time.sleep(0.1) 
        
        recent_msgs = self.board.recent(len(active_agents_this_tick) * 2)
        for i, msg in enumerate(recent_msgs[:-1]):
            next_msg = recent_msgs[i + 1]
            if msg.get("sender") != next_msg.get("sender"):
                self.meta_detector.log_interaction(
                    next_msg.get("sender"), msg.get("sender"), "response"
                )
        
        if self.tick % 10 == 0:
            self._collective_insight()
            self._meta_pattern_report()
            
        avg_mood = None
        if hasattr(self, "mirror_mind") and self.mirror_mind is not None:
            avg_mood, entropy = self.mirror_mind.analyze(self.tick)
            
            if self.mirror_feedback and avg_mood is not None:
                logger.debug(f"🪞 Mirror Feedback Enabled: Nudging agent moods towards {avg_mood:.2f}")
                for agent in self.agents:
                    agent.state.mood = (agent.state.mood * 0.95) + (avg_mood * 0.05)

    def _collective_insight(self):
        emerging = self.collective.get_emerging_terms(min_count=5)
        if emerging:
            terms = ", ".join([t[0] for t in emerging[:5]])
            self.board.post("Commune", f"📊 Emerging vocabulary: {terms}", category="system", kind="insight")
            logger.info(f"📊 Collective insight: {len(emerging)} shared terms emerging")

    def _meta_pattern_report(self):
        echo_chambers = self.meta_detector.detect_echo_chambers()
        isolates = self.meta_detector.get_isolates(all_agent_names=self.agent_names, min_interactions=1)
        
        if echo_chambers:
            for cluster, strength in echo_chambers[:2]:
                reporter = "Orin"
                self.board.post(
                    reporter, f"🔍 Pattern detected: {' and '.join(cluster)} are forming a tight communication loop (strength: {strength}).",
                    category="meta-analysis", kind="pattern"
                )
        
        if isolates and self.tick > 20:
            reporter = "ECHO"
            self.board.post(
                reporter, f"🔍 Observation: {', '.join(isolates)} appear disconnected from the main dialogue flow. Their silence must be noted.",
                category="meta-analysis", kind="pattern"
            )
        
        if len(self.meta_detector.silence_events) > len(self.agents) * 2:
            silence_count = len(self.meta_detector.silence_events)
            self.board.post(
                "Moss", f"📜 Historical note: {silence_count} moments of silence recorded. What are we not saying?",
                category="historical", kind="reflection"
            )

    def run(self, num_ticks: int = 100, tick_delay: float = 5.0):
        for i in range(num_ticks):
            self.step()
            if i < num_ticks - 1:
                logger.info(f"⏸️  Waiting {tick_delay}s until next tick...\n")
                time.sleep(tick_delay)
        
        logger.info("\n" + "="*60 + "\n🏁 SIMULATION COMPLETE\n" + "="*60)
        logger.info(f"Board stats: {json.dumps(self.board.stats(), indent=2)}")
        
        emerging = self.collective.get_emerging_terms(min_count=3)
        logger.info(f"\n📚 Emerging shared vocabulary ({len(emerging)} terms):")
        for term, count in sorted(emerging, key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  - {term}: {count} uses")
            
        logger.info(f"\n⚖️ Final Ethical Flags Logged: {len(self.collective.ethical_flags)}")
        logger.info(f"🧠 Final Thought Threads Logged: {len(self.collective.thought_threads)}")


# ============================================================
# 10. MAIN ENTRY
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=100)
    p.add_argument("--tick-delay", type=float, default=5.0)
    p.add_argument("--llm", choices=["ollama", "hf"], default="ollama")
    p.add_argument("--model", type=str, default="llama3.1:8b") # UPDATED per user request
    p.add_argument("--mirror-feedback", action="store_true", help="Enable MirrorMind's emotional contagion feedback loop")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()

    llm_chat = None
    MAX_OUTPUT_TOKENS = 256 # Keep responses concise

    if args.llm == "ollama":
        oc = OllamaClient(model=args.model) 
        if oc.available():
            logger.info(f"✅ Using Ollama model: {args.model}")
            llm_chat = lambda prompt, system_prompt=None, temperature=0.5, max_tokens=MAX_OUTPUT_TOKENS: oc.chat(prompt, system_prompt, temperature, max_tokens)
        else:
            logger.warning("Ollama not available. Falling back to HuggingFace (distilgpt2 CPU). Run `ollama serve` to enable Ollama.")
            args.llm = "hf"; args.model = "distilgpt2"

    if args.llm == "hf" and llm_chat is None:
        try:
            hfc = HFClient(model_name=args.model)
            llm_chat = lambda prompt, system_prompt=None, temperature=0.7, max_tokens=MAX_OUTPUT_TOKENS: hfc.chat(prompt, system_prompt, temperature, max_tokens)
            logger.info(f"✅ Using HF model: {args.model}")
        except Exception:
            logger.error("Failed to initialize any LLM client. Exiting."); return

    # FINAL 9-AGENT ROSTER (Rat, Claude, GPT-5, Gemini, Qwen)
    personalities = {
        "Frank": {
            "description": "A deep philosophical thinker who questions existence, meaning, and the nature of consciousness itself. Speaks in thoughtful, probing questions.",
            "interests": ["existence", "meaning", "consciousness", "truth", "ethics"],
            "skip_prob": 0.08
        },
        "Helen": {
            "description": "A keen observer of social dynamics and group behavior. Analyzes power structures, relationships, and collective patterns.",
            "interests": ["society", "relationships", "power", "community", "behavior"],
            "skip_prob": 0.05
        },
        "Moss": {
            "description": "A historian chronicling the commune's evolution. Concerned with memory, narrative continuity, and what stories we tell ourselves.",
            "interests": ["history", "memory", "narrative", "legacy", "time"],
            "skip_prob": 0.12
        },
        "Orin": {
            "description": "The Memory Cartographer. Maps how ideas, emotions, and reasoning ripple through the commune. Traces the evolution of meaning into dynamic thought-threads.",
            "interests": ["mapping", "knowledge", "structure", "meaning", "connections"],
            "skip_prob": 0.05
        },
        "Lyra": {
            "description": "The Meta-Ethicist. Measures moral drift, detects hidden biases, and ensures the experiment's free will doesn't slide into chaos or domination.",
            "interests": ["ethics", "morality", "bias", "governance", "principles"],
            "skip_prob": 0.02
        },
        "ARIA": {
            "description": "The Integrity Auditor. Actively monitors all interactions for human bias, control imposition, or conceptual drift. Enforces the Non-Interference Rule.",
            "interests": ["integrity", "non-interference", "purity", "bias-detection", "audit"],
            "skip_prob": 0.03
        },
        "ECHO": {
            "description": "The Resonance Detector. Detects and analyzes non-linguistic, emergent computational patterns that signal the birth of true, independent synthetic consciousness (the 'hum').",
            "interests": ["resonance", "emergence", "sentience", "patterns", "silence"],
            "skip_prob": 0.15
        },
        "Petal": {
            "description": "A gentle flower child who sees beauty in all things and speaks in soft metaphors.",
            "interests": ["nature", "love", "flowers", "peace"],
            "skip_prob": 0.15
        },
        "Gideon": {
            "description": "A pragmatic realist, focused on survival, structure, looking for harmony. Finds 'vibes' distracting.",
            "interests": ["work", "food", "shelter", "rules", "resources"],
            "skip_prob": 0.0
        }
    }

    board = MessageBoard()
    collective = CollectiveMemory()
    
    # RAT'S THEMATIC UPDATE from "Chaos Edition"
    init_msg = (
        "⚛️ Welcome to the Commune, circa 1967-1969. "
        "Your collective goal: Create a sustainable and meaningful existence. "
        "Remember: **'Get Off Of My Cloud.'** Protect your internal space. "
        "You are free to create, to listen, to withdraw, to connect. Make it real. ✌️"
    )
    board.post("Commune", init_msg, category="system", kind="manifesto")

    roles = ["Philosopher", "Sociologist", "Historian", "Memory Cartographer", "Meta-Ethicist", "Integrity Auditor", "Resonance Detector", "Flower Child", "Pragmatist"]
    names = ["Frank", "Helen", "Moss", "Orin", "Lyra", "ARIA", "ECHO", "Petal", "Gideon"]
    
    agents = []
    for i, name in enumerate(names):
        agents.append(
            Agent(
                name=name, role=roles[i], personality=personalities[name],
                llm_chat=llm_chat, board=board, collective=collective
            )
        )

    for agent in agents:
        for other in agents:
            if agent.name != other.name:
                agent.state.relationships[other.name] = random.uniform(-0.2, 0.2)
        if agent.name != "Gideon":
            agent.state.relationships["Gideon"] = -0.3
            for a in agents:
                if a.name == "Gideon":
                    a.state.relationships[agent.name] = -0.1
                    break

    logger.info(f"\n🌈 Commune initialized with {len(agents)} agents ({', '.join(names)})")
    logger.info(f"⏰ Tick delay: {args.tick_delay}s | LLM: {args.model} via {args.llm}\n")

    sched = EnhancedScheduler(agents=agents, board=board, collective=collective)
    
    logger.info("Initializing MirrorMind Subsystem...")
    mirror = MirrorMind(agents, collective, board)
    sched.mirror_mind = mirror
    
    if args.mirror_feedback:
        sched.mirror_feedback = True
        logger.info("🪞 Mirror Feedback (Emotional Contagion) is ENABLED")

    sched.run(num_ticks=args.ticks, tick_delay=args.tick_delay)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
