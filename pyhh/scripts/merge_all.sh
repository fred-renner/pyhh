# samples to merge
declare -a samples=(
    "mc20_ttbar"
    "run2"
    "mc20_dijet"
    "mc20_SM"
    "mc20_k2v0"
)

## now loop through the above array
for i in "${samples[@]}"; do
    echo "$i"
    /lustre/fs22/group/atlas/freder/hh/pyhh/pyhh/main.py --merge --sample $i &

    # or do whatever with individual element of the array
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also
