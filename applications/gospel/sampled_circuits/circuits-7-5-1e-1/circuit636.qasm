OPENQASM 2.0;
include "qelib1.inc";
qreg q637[7];
cx q637[4],q637[5];
cx q637[4],q637[3];
cx q637[2],q637[3];
cx q637[2],q637[1];
cx q637[1],q637[0];
