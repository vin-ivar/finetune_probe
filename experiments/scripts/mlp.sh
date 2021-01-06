cd ../..
for lang in en_ewt; do 
	for layer in 0 0-1 0-2 0-3; do 
		./submit.sh $lang kill.mlp_norm.$layer none component
	done
done
