# samples to submit
declare -a samples=(
    "mc20_ttbar"
    "run2"
    "mc20_dijet"
    "mc20_l1cvv1cv1"
)

# copy and run from submit folder
rsync -r --exclude=.git /lustre/fs22/group/atlas/freder/hh/hh-analysis /lustre/fs22/group/atlas/freder/hh/submit/

# now loop through the above array
for i in "${samples[@]}"; do
    echo "$i"
    /lustre/fs22/group/atlas/freder/hh/hh-analysis/makeSubmitFile.py --sample $i
    cd /lustre/fs22/group/atlas/freder/hh/submit/$i
    condor_submit /lustre/fs22/group/atlas/freder/hh/submit/HistFill_$i.sub

    # or do whatever with individual element of the array
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also
