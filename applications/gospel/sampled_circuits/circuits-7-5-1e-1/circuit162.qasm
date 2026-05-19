OPENQASM 2.0;
include "qelib1.inc";
qreg q163[7];
cx q163[4],q163[5];
cx q163[3],q163[4];
cx q163[3],q163[2];
cx q163[1],q163[2];
cx q163[0],q163[1];
