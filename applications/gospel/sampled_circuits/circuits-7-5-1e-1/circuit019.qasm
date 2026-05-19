OPENQASM 2.0;
include "qelib1.inc";
qreg q20[7];
cx q20[4],q20[5];
cx q20[3],q20[4];
cx q20[2],q20[3];
cx q20[2],q20[1];
cx q20[0],q20[1];
