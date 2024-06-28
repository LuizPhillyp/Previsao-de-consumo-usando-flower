echo "Starting server"

python3 server.py &
sleep 10  # Sleep for 10s to give the server enough time to start and download the dataset

for i in $(seq 1 19); do
    echo "Starting client $i"
    python3 client.py --client-id=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait