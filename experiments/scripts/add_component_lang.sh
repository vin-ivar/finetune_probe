cd ../..
for lang in zh_gsd ja_gsd ar_padt hi_hdtb; do 
	for component in keys queries; do 
		for layer in 0 0-1 0-2 0-6 0-7 0-10 0-11; do 
			./submit.sh $lang kill.$component.$layer none component
		done
	done
done
