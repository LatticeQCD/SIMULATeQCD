#!/bin/bash

ml=$1
ms=$2
Nfl=$3
Nfs=$4
Npf=$5

_ml=${ml/*./}
_ms=${ms/*./}
_Nfl=${Nfl/./}
_Nfs=${Nfs/./}
_Npf=${Npf/./}


if [ $(echo "abs(8-$Nf) < 0.01"| bc) ]
then
_Nf=8
fi

cat > in.files/ml${_ml}ms${_ms}Nfl${_Nfl}Nfs${_Nfs}Npf${_Npf}  << EOF
2 

$(echo "scale=5;($Nfl+$Nfs)/$Npf" | bc)
0
0      
$ms    
14       
12       
0$(echo "scale=10;$ms*$ms/2" | bc)
5.0      
50     

$(echo "scale=5;$Nfl/$Npf" | bc)  
-$(echo "scale=5;$Nfl/$Npf" | bc)     
$ms      
$ml    
14       
12       
0$(echo "scale=10;$ml*$ml/2" | bc)
5.0      
160    
EOF

./poly4 in.files/ml${_ml}ms${_ms}Nfl${_Nfl}Nfs${_Nfs}Npf${_Npf} > out.files/rat.out_ml${_ml}ms${_ms}Nfl${_Nfl}Nfs${_Nfs}Npf${_Npf}
