#!/bin/bash
for((i=1;i<81;i++))
do
labelme_json_to_dataset ${i}.json
done