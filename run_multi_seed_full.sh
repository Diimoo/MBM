#!/bin/bash
# Run full PPO training on 5 seeds

for seed in 100 101 102 103 104; do
    echo "=========================================="
    echo "SEED $seed - Starting full PPO training"
    echo "=========================================="
    
    # Remove old checkpoint to force fresh training
    rm -f brain_vectorized_best.pth
    
    # Run full PPO for 50k steps
    python3 train_vectorized.py \
        --seed $seed \
        --total_steps 50000 \
        2>&1 | tee "logs/seed_${seed}_full.log"
    
    # Save final checkpoint with seed name
    if [ -f brain_vectorized_best.pth ]; then
        cp brain_vectorized_best.pth "checkpoints/brain_seed${seed}_best.pth"
    fi
    
    echo ""
done

echo "=========================================="
echo "Multi-seed training complete!"
echo "=========================================="

# Quick summary
echo ""
echo "SUMMARY:"
for seed in 100 101 102 103 104; do
    if [ -f "logs/seed_${seed}_full.log" ]; then
        best_sr=$(grep "New Best SR" "logs/seed_${seed}_full.log" | tail -1 | grep -oP 'SR: [0-9.]+' | grep -oP '[0-9.]+')
        if [ -n "$best_sr" ]; then
            echo "  Seed $seed: Best SR = $best_sr"
        else
            # Try alternate format
            best_sr=$(grep "Eval SR:" "logs/seed_${seed}_full.log" | tail -1 | grep -oP '[0-9.]+$')
            echo "  Seed $seed: Final SR = $best_sr"
        fi
    fi
done
