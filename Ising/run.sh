for L in 8 ; do
  echo "Iniciando L=$L..."
  ./ising $L > ising_L${L}.log 2>&1 &
done
wait
echo "Todas as runs finalizadas."