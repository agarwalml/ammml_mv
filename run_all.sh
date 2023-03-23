#!/usr/bin/env bash

for ms in 000000 100000 010000 001000 000100 000010 000001 111111
do
    while read -r url
    do
        id=${ms}_${url: -11}
        echo $id
        echo -e "#!/usr/bin/env bash\n\nconda activate mv\npython baseline.py $url /results/sbenoit/ammml_mv/$id >& $id.out\nconda activate ffmpeg\nffmpeg -f concat -i concat.txt -i audio.mp3 -map 0:v -map 1:a -c:v libx264 -vsync vfr -vf mpdecimate -pix_fmt yuv420p -c:a copy -shortest -y video.mp4\n" > $id.sh
        chmod +x $id.sh
        sbatch -p gpu_low --gres=gpu:1 --mem=16G -x compute-1-17 $id.sh
    done < urls.txt
done

