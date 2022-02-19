# 
# isParamFileInUse.bash                                                               
# 
# D. Clarke 
# 
# Script to check for any unused .param files. Call this from within the
# current folder. 
#

paramFolder='../parameter/'
srcFolder='../src/'

for f in ${paramFolder}*; do

  if [ -d $f ]; then
    cd $f
    for g in *.param; do
      echo
      echo "---"$g"---"
      grep -r "$g" "../"${srcFolder}*
    done
    cd ..
  fi

  ext=${f##*.}
  if [ ${ext} = 'param' ]; then 
    echo
    echo "---"$f"---"
    grep -r "${f##*/}" ${srcFolder}* 
  fi     

done
