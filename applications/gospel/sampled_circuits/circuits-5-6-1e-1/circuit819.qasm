OPENQASM 2.0;
include "qelib1.inc";
qreg q820[5];
cx q820[3],q820[4];
cx q820[2],q820[3];
cx q820[1],q820[2];
cx q820[1],q820[0];
rx(pi/4) q820[1];
