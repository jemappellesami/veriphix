OPENQASM 2.0;
include "qelib1.inc";
qreg q978[3];
rx(3*pi/2) q978[2];
rz(pi/2) q978[2];
cx q978[1],q978[2];
cx q978[1],q978[0];
rx(pi/4) q978[1];
