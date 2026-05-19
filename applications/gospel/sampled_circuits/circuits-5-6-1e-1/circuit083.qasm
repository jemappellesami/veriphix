OPENQASM 2.0;
include "qelib1.inc";
qreg q84[5];
rx(7*pi/4) q84[3];
cx q84[3],q84[2];
cx q84[1],q84[2];
cx q84[0],q84[1];
rx(pi/4) q84[1];
