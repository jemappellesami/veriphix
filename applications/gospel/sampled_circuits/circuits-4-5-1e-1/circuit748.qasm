OPENQASM 2.0;
include "qelib1.inc";
qreg q749[4];
rz(7*pi/4) q749[3];
rx(5*pi/4) q749[3];
cx q749[3],q749[2];
cx q749[2],q749[1];
cx q749[1],q749[0];
