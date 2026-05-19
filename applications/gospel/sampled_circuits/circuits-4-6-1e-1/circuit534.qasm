OPENQASM 2.0;
include "qelib1.inc";
qreg q535[4];
rz(pi/2) q535[3];
cx q535[2],q535[3];
cx q535[2],q535[1];
cx q535[0],q535[1];
rx(pi/4) q535[1];
