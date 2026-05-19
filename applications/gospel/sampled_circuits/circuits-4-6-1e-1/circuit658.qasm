OPENQASM 2.0;
include "qelib1.inc";
qreg q659[4];
rx(5*pi/4) q659[3];
cx q659[2],q659[3];
cx q659[2],q659[1];
cx q659[0],q659[1];
rx(pi/4) q659[1];
