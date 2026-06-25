for L in 34 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100; do
  echo "Iniciando L=$L..."
  ./ising $L > ising_L${L}.log 2>&1 &
done
wait
echo "Todas as runs finalizadas."