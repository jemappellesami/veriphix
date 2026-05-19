OPENQASM 2.0;
include "qelib1.inc";
qreg q740[7];
cx q740[5],q740[4];
cx q740[4],q740[3];
cx q740[2],q740[3];
cx q740[2],q740[1];
cx q740[0],q740[1];
