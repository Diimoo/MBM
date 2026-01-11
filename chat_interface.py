#!/usr/bin/env python3
"""
Interactive chat interface for German Q&A model.
Supports context retrieval for factual answers.
"""

import torch
from pathlib import Path
import readline  # For better input handling

from hierarchical_german_phase6_qa import (
    HierarchicalGermanQA, VOCAB_SIZE, text_to_indices
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use large corpus checkpoint (no RAG - model learns directly)
CHECKPOINT_PATH = Path("checkpoints/phase6_qa_large.pth")
FALLBACK_CHECKPOINT = Path("checkpoints/phase6_qa_best.pth")


class GermanQAChat:
    """Interactive German Q&A chatbot."""
    
    def __init__(self):
        self.model = None
        self.retriever = None
        self.history = []
        self.temperature = 0.7
    
    def load(self):
        """Load model and retriever."""
        print("🔄 Loading model...")
        
        # Load model
        self.model = HierarchicalGermanQA(VOCAB_SIZE).to(DEVICE)
        
        # Try retrieval checkpoint first, then fallback
        checkpoint_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else FALLBACK_CHECKPOINT
        
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            # Handle both full checkpoint and state_dict only
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            state_dict = {k: v for k, v in state_dict.items() if 'causal_mask' not in k}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"   ✅ Loaded from {checkpoint_path}")
        else:
            print("   ⚠️ No checkpoint found, using random weights")
        
        self.model.eval()
        
        # No RAG - model learns directly
        self.retriever = None
        
        print("\n✅ Ready!")
    
    def retrieve_context(self, question):
        """Retrieve relevant context for a question."""
        if not self.retriever:
            return None, 0.0
        
        results = self.retriever.retrieve(question, top_k=1)
        if results and results[0][0] > 0.1:
            return results[0][1], results[0][0]
        return None, 0.0
    
    def answer(self, question):
        """Generate answer to a question."""
        # Retrieve context
        context, score = self.retrieve_context(question)
        
        # Generate answer
        try:
            response = self.model.answer_question(question, temperature=self.temperature)
            
            # Clean up response
            response = response.strip()
            if response.startswith("<A>"):
                response = response[3:]
            if "<EOS>" in response:
                response = response[:response.index("<EOS>")]
            if "<PAD>" in response:
                response = response[:response.index("<PAD>")]
            
            return response.strip(), context, score
        except Exception as e:
            return f"[Fehler: {e}]", context, score
    
    def chat(self):
        """Run interactive chat loop."""
        print("\n" + "=" * 60)
        print("🇩🇪 German Q&A Chat")
        print("=" * 60)
        print("\nBefehle:")
        print("  /quit     - Beenden")
        print("  /temp N   - Temperatur setzen (0.1-2.0)")
        print("  /history  - Verlauf anzeigen")
        print("  /clear    - Verlauf löschen")
        print("  /help     - Diese Hilfe anzeigen")
        print("\n" + "-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 Du: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nAuf Wiedersehen! 👋")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower()
                
                if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                    print("\nAuf Wiedersehen! 👋")
                    break
                
                elif cmd.startswith("/temp"):
                    try:
                        temp = float(cmd.split()[1])
                        self.temperature = max(0.1, min(2.0, temp))
                        print(f"   Temperatur: {self.temperature}")
                    except:
                        print(f"   Aktuelle Temperatur: {self.temperature}")
                
                elif cmd == "/history":
                    if self.history:
                        print("\n📜 Verlauf:")
                        for i, (q, a) in enumerate(self.history[-10:], 1):
                            print(f"   {i}. Q: {q[:50]}...")
                            print(f"      A: {a[:50]}...")
                    else:
                        print("   Kein Verlauf vorhanden.")
                
                elif cmd == "/clear":
                    self.history = []
                    print("   Verlauf gelöscht.")
                
                elif cmd == "/help":
                    print("\nBefehle:")
                    print("  /quit     - Beenden")
                    print("  /temp N   - Temperatur setzen")
                    print("  /history  - Verlauf anzeigen")
                    print("  /clear    - Verlauf löschen")
                
                else:
                    print(f"   Unbekannter Befehl: {cmd}")
                
                continue
            
            # Ensure question ends with ?
            if not user_input.endswith("?"):
                user_input += "?"
            
            # Generate answer
            print("\n🤖 Assistent: ", end="", flush=True)
            answer, context, score = self.answer(user_input)
            print(answer)
            
            # Show retrieval info if context was used
            if context and score > 0.15:
                print(f"   📚 (Kontext: {context[:60]}... Relevanz: {score:.2f})")
            
            # Save to history
            self.history.append((user_input, answer))


def main():
    """Main entry point."""
    print("=" * 60)
    print("🇩🇪 German Q&A Chat Interface")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    chat = GermanQAChat()
    chat.load()
    chat.chat()


if __name__ == "__main__":
    main()
