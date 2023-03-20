#!/usr/bin/env bash

while read -r url
do
    id=${url: -11}
    echo $id
    echo -e "#!/usr/bin/env bash\n\nconda activate mv\npython baseline.py $url /results/sbenoit/ammml_mv/$id >& $id.out\n" > $id.sh
    chmod +x $id.sh
    sbatch -p gpu_low --gres=gpu:1 --mem=16G $id.sh
done < urls.txt

