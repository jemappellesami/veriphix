OPENQASM 2.0;
include "qelib1.inc";
qreg q535[4];
cx q535[2],q535[3];
rz(3*pi/2) q535[3];
cx q535[3],q535[2];
cx q535[2],q535[1];
cx q535[1],q535[0];
