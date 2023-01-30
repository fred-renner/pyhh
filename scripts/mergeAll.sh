# samples to merge
declare -a samples=(
    "mc20_ttbar"
    "run2"
    "mc20_dijet"
    "mc20_l1cvv1cv1"
)

## now loop through the above array
for i in "${samples[@]}"; do
    echo "$i"
    /lustre/fs22/group/atlas/freder/hh/hh-analysis/Merger.py --sample $i &

    # or do whatever with individual element of the array
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also
