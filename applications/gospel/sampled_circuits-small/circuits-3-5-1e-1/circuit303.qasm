OPENQASM 2.0;
include "qelib1.inc";
qreg q304[3];
rz(7*pi/4) q304[2];
cx q304[1],q304[2];
cx q304[1],q304[0];
