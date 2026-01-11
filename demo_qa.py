#!/usr/bin/env python3
import torch
import sys
import os
import random
from hierarchical_german_phase8 import HierarchicalGermanPhase8, text_to_indices, indices_to_text, idx_to_char

def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lade Modell auf {device}...")
    
    model = HierarchicalGermanPhase8().to(device)
    checkpoint_path = "checkpoints/phase8_discourse_best.pth"
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/phase7_ach_best.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Fehler: Checkpoint nicht gefunden.")
            return
        print(f"Hinweis: Lade Phase 7 Checkpoint (Phase 8 fehlt).")

    print(f"Lade Gewichte von {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print("\n" + "="*60)
    print("INTERAKTIVER HIERARCHISCHER DEUTSCH-QA DEMO (Phase 8: Diskurs)")
    print("="*60)
    print("Das Modell behält nun den Kontext über mehrere Sätze bei.")
    print("Befehle:")
    print("- 'exit': Beenden")
    print("- 'reset': Diskurs-Speicher leeren")
    print("- 'correct <Satz>': Das Modell mit einem korrekten Satz trainieren")
    print("- 'explain': Die Hebbian-Assoziationen für den letzten Satz erklären")
    print("- 'info': Brain-Status und Hierarchie-Gewichte anzeigen")
    
    discourse_state = None
    memory_buffer = None
    last_word_emb = None
    
    try:
        while True:
            user_input = input("\nDu: ").strip()
            if user_input.lower() in ['exit', 'quit', 'beenden']:
                break
            
            if user_input.lower() == 'reset':
                discourse_state = None
                memory_buffer = None
                last_word_emb = None
                print("[System] Diskurs-Speicher wurde geleert.")
                continue
            
            if user_input.lower() == 'info':
                temp = model.serotonin.get_temperature()
                weights, ach, precision = model.acetylcholine([])
                reward = model.dopamine.reward_baseline.item()
                print(f"[Brain State] ACh (Unsicherheit): {ach:.2f}, 5-HT (Mood): {temp:.2f}, DA (Reward): {reward:.2f}")
                print(f"[Hierarchy Weights] L0(Char): {weights[0]:.2f}, L1(Syl): {weights[1]:.2f}, L2(Morph): {weights[2]:.2f}, L3(Word): {weights[3]:.2f}, L4(Phrase): {weights[4]:.2f}, L5(Sent): {weights[5]:.2f}")
                continue

            if user_input.lower() == 'explain':
                if last_word_emb is None:
                    print("[System] Bitte gib zuerst einen Satz ein.")
                    continue
                
                # Visualize Hebbian associations from word to sentence level
                with torch.no_grad():
                    # Association matrix is [d_word, d_sent]
                    assoc = model.hebbian.association_matrix
                    # Get strongest neurons in sentence embedding triggered by last word
                    # last_word_emb: [1, d_word]
                    trigger = last_word_emb.mean(dim=0)
                    impact = torch.matmul(trigger, assoc)
                    top_impact = impact.topk(5)
                    print(f"[Explain] Das Wort-Muster hat die stärksten Assoziationen zu diesen Diskurs-Dimensionen:")
                    print(f"  Assoziations-Stärke: {top_impact.values.tolist()}")
                continue

            if user_input.lower().startswith('correct '):
                correction = user_input[8:].strip()
                if correction:
                    msg = model.learn_from_correction(correction)
                    print(f"[System] {msg} Hebbian-Assoziationen wurden für '{correction}' gestärkt.")
                continue
                
            if not user_input:
                continue
                
            is_q = model.is_question(user_input, discourse_state=discourse_state, memory_buffer=memory_buffer)
            
            if is_q:
                print(f"[System] Frage erkannt.")
                answer, discourse_state, memory_buffer = model.answer_question(user_input, discourse_state=discourse_state, memory_buffer=memory_buffer)
                print(f"Bot: {answer}")
            else:
                print(f"[System] Aussage erkannt.")
                question, discourse_state, memory_buffer = model.ask_question(user_input, discourse_state=discourse_state, memory_buffer=memory_buffer)
                # Store word embedding for explainability
                with torch.no_grad():
                    _, word_emb, _, _, _ = model.encode(torch.tensor([text_to_indices(user_input)], device=device))
                    last_word_emb = word_emb.mean(dim=1)
                print(f"Bot fragt: {question}")
                
    except KeyboardInterrupt:
        print("\nDemo beendet.")

if __name__ == "__main__":
    run_demo()
