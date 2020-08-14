#! /bin/bash

# TODO: generic script for both zsh && bash
NUM_TEST=5
DATA_PATH=/PATH/TO/EUROC/DATASET
V1_SEQUENCES=('V1_01_easy' 'V1_02_medium' 'V1_03_difficult')
# V1_SEQUENCES=('V1_01_easy')
V2_SEQUENCES=('V2_01_easy' 'V2_02_medium')
# V2_SEQUENCES=('V2_02_medium')

trap ctrl_c INT

function ctrl_c() {
	kill %1
	exit
}

function srcc() {
        prefix=''
        if [ "$#" -eq 1 ]; then
                prefix="_$1"
                print $prefix
        fi

        cwd=`pwd`
        cdir=`pwd`
        while [ $cdir != $HOME ]; do
                dir_to_check="$cdir/devel$prefix"
                if [ -d $dir_to_check ]; then
                        source "$dir_to_check/setup.bash";
                        echo "source $dir_to_check/setup.bash";
                        break;
                fi
                cd ..
                cdir=`pwd`
        done
        if [ $cdir = $HOME ]; then
                echo "failed to find a catkin... "
        fi

        cd $cwd
}


rm -rf $(rospack find gmmloc_ros)/expr/
mkdir -p $(rospack find gmmloc_ros)/expr/

srcc

roscore &

sleep 2

rosparam set /gmmloc/vocabulary_path $(rospack find gmmloc_ros)/voc/ORBvoc.bin
rosparam set /gmmloc/output_path $(rospack find gmmloc_ros)/expr/

rosparam set /gmmloc/gmm_path $(rospack find gmmloc_ros)/data/map/v1.gmm
rosparam set /gmmloc/rect_config $(rospack find gmmloc_ros)/cfg/euroc_rect.yaml

# rosparam set /gmmloc/online True

rosparam load $(rospack find gmmloc_ros)/cfg/v1.yaml /gmmloc
expr_path=$(rospack find gmmloc_ros)/expr/

for seq in ${V1_SEQUENCES[*]}
do
	rosparam set /gmmloc/data_path "$DATA_PATH/$seq/mav0/"
	rosparam set /gmmloc/gt_path $(rospack find gmmloc_ros)/data/gt_sync/$seq.txt

	for (( i=1; i<=$NUM_TEST; i++ ))
	do
		echo 
		echo '#####################################################################' 
		echo "$seq, $i/$NUM_TEST"

		rosrun gmmloc gmmloc_node __name:=gmmloc -alsologtostderr -colorlogtostderr --minloglevel=2

		mv $expr_path/traj_est.txt $expr_path/$seq$i.txt
	done
done

rosparam set /gmmloc/gmm_path $(rospack find gmmloc_ros)/data/map/v2.gmm
rosparam set /gmmloc/rect_config $(rospack find gmmloc_ros)/cfg/euroc_rect.yaml

rosparam load $(rospack find gmmloc_ros)/cfg/v2.yaml /gmmloc
expr_path=$(rospack find gmmloc_ros)

for seq in ${V2_SEQUENCES[*]}
do
	rosparam set /gmmloc/data_path "$DATA_PATH/$seq/mav0/"
	rosparam set /gmmloc/gt_path $(rospack find gmmloc_ros)/data/gt_sync/$seq.txt

	for (( i=1; i<=$NUM_TEST; i++ ))
	do
		echo 
		echo '#####################################################################'
		echo "$seq, $i/$NUM_TEST"

		rosrun gmmloc gmmloc_node __name:=gmmloc -alsologtostderr -colorlogtostderr --minloglevel=2

		mv $expr_path/expr/traj_est.txt $expr_path/expr/$seq$i.txt
	done
done

python3 scripts/evo_euroc.py --pkg_path $(rospack find gmmloc_ros)

kill %1

