cd ../..
for lang in en_ewt it_isdt sv_talbanken; do 
	for component in dense; do 
		for layer in 0 0-1 0-2 0-6 0-7 0-10 0-11; do 
			./submit.sh $lang kill.$component.$layer none component
		done
	done
done
