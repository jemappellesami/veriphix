OPENQASM 2.0;
include "qelib1.inc";
qreg q568[3];
rx(5*pi/4) q568[0];
rz(pi) q568[0];
rx(pi/2) q568[2];
cx q568[1],q568[2];
cx q568[0],q568[1];
