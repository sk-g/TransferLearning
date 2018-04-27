start=$SECONDS
echo "ResNet50"
python3 transfer_learning.py --model=ResNet50

echo "VGG16"
python3 transfer_learning.py --model=VGG16

echo "VGG19"
python3 transfer_learning.py --model=VGG19

echo "InceptionResNetV2"
python3 transfer_learning.py --model=InceptionResNetV2

echo "Xception"
python3 transfer_learning.py --model=Xception

echo "InceptionV3"
python3 transfer_learning.py --model=InceptionV3

echo "DenseNet"
python3 transfer_learning.py --model=DenseNet

duration=$((SECONDS-start))

if (( $duration > 3600 )) ; then
    let "hours=duration/3600"
    let "minutes=(duration%3600)/60"
    let "seconds=(duration%3600)%60"
    echo "Completed in $hours hour(s), $minutes minute(s) and $seconds second(s)" 
elif (( $duration > 60 )) ; then
    let "minutes=(duration%3600)/60"
    let "seconds=(duration%3600)%60"
    echo "Completed in $minutes minute(s) and $seconds second(s)"
else
    echo "Completed in $duration seconds"
fi
