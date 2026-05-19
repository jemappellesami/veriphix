OPENQASM 2.0;
include "qelib1.inc";
qreg q715[4];
rx(pi/4) q715[3];
rz(5*pi/4) q715[3];
cx q715[3],q715[2];
cx q715[2],q715[1];
cx q715[1],q715[0];
