# samples to submit
declare -a samples=(
    "mc20_SM"
    # "mc20_k2v0"
    # "mc20_ttbar"
    # "mc20_dijet"
    # "run2"
)

# copy and run from submit folder
rm -rf /lustre/fs22/group/atlas/freder/hh/submit/pyhh/
rsync -r --exclude=.git /lustre/fs22/group/atlas/freder/hh/pyhh /lustre/fs22/group/atlas/freder/hh/submit/

# now loop through the above array
for i in "${samples[@]}"; do

    /lustre/fs22/group/atlas/freder/hh/pyhh/pyhh/scripts/make_histfill_sub.py --sample $i
    mkdir /lustre/fs22/group/atlas/freder/hh/submit/$i -p
    cd /lustre/fs22/group/atlas/freder/hh/submit/$i
    condor_submit /lustre/fs22/group/atlas/freder/hh/submit/histfill_$i.sub

    # or do whatever with individual element of the array
done

# You can access them using echo "${arr[0]}", "${arr[1]}" also