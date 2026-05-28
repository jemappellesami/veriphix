OPENQASM 2.0;
include "qelib1.inc";
qreg q273[3];
rx(3*pi/4) q273[2];
rz(7*pi/4) q273[2];
cx q273[2],q273[1];
cx q273[1],q273[0];
rx(pi/4) q273[1];
