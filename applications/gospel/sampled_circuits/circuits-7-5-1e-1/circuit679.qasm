OPENQASM 2.0;
include "qelib1.inc";
qreg q680[7];
cx q680[4],q680[5];
cx q680[4],q680[3];
cx q680[2],q680[3];
cx q680[1],q680[2];
cx q680[0],q680[1];
