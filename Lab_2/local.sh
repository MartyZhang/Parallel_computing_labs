ip=$1
ssh "ubuntu@$ip" -i /mnt/c/Users/micha/.ssh/id_rsa "nohup mpirun -np 8 python ~/Parallel_computing_labs/Lab_2/grid_512_512.py 2000 > ~/output.txt 2>&1 &"

scp -i /mnt/c/Users/micha/.ssh/id_rsa "ubuntu@$1:/home/ubuntu/output.txt" .