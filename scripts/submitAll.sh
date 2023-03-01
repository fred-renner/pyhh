# samples to submit
declare -a samples=(
    "mc20_ttbar"
    "run2"
    "mc20_dijet"
    "mc20_SM"
)

# copy and run from submit folder
rm -rf /lustre/fs22/group/atlas/freder/hh/submit/pyhh/
rsync -r --exclude=.git /lustre/fs22/group/atlas/freder/hh/pyhh /lustre/fs22/group/atlas/freder/hh/submit/

# now loop through the above array
for i in "${samples[@]}"; do
    echo "$i"
    /lustre/fs22/group/atlas/freder/hh/pyhh/scripts/makeSubmitFile.py --sample $i
    cd /lustre/fs22/group/atlas/freder/hh/submit/$i
    condor_submit /lustre/fs22/group/atlas/freder/hh/submit/HistFill_$i.sub

    # or do whatever with individual element of the array
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also
