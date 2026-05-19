OPENQASM 2.0;
include "qelib1.inc";
qreg q235[7];
cx q235[4],q235[5];
cx q235[3],q235[4];
cx q235[2],q235[3];
cx q235[1],q235[2];
cx q235[0],q235[1];
