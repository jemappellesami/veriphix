OPENQASM 2.0;
include "qelib1.inc";
qreg q184[3];
rz(5*pi/4) q184[2];
cx q184[1],q184[2];
cx q184[1],q184[0];
rx(pi/4) q184[1];
