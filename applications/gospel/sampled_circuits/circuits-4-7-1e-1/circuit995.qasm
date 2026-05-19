OPENQASM 2.0;
include "qelib1.inc";
qreg q996[4];
cx q996[3],q996[2];
cx q996[2],q996[3];
rz(5*pi/4) q996[2];
cx q996[2],q996[1];
cx q996[1],q996[0];
