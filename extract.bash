python extract_reset.py &
sleep 10.0 
for i in `seq 1 9`;
do
  echo worker $i
  # on cloud:
  python extract.py --port $i &
  # on macbook for debugging:
  #python extract.py &
  #sleep 1.0
done
