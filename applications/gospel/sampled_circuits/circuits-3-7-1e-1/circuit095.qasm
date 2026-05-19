OPENQASM 2.0;
include "qelib1.inc";
qreg q96[3];
rx(pi/2) q96[2];
rz(3*pi/2) q96[2];
rx(pi/2) q96[2];
cx q96[2],q96[1];
cx q96[1],q96[0];
