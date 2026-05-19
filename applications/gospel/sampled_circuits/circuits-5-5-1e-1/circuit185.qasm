OPENQASM 2.0;
include "qelib1.inc";
qreg q186[5];
cx q186[4],q186[3];
cx q186[3],q186[2];
cx q186[2],q186[1];
cx q186[1],q186[0];
