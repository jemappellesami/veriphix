OPENQASM 2.0;
include "qelib1.inc";
qreg q249[6];
rx(5*pi/4) q249[0];
cx q249[0],q249[1];
cx q249[1],q249[2];
cx q249[0],q249[1];
