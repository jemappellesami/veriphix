OPENQASM 2.0;
include "qelib1.inc";
qreg q84[5];
cx q84[1],q84[2];
cx q84[3],q84[4];
cx q84[2],q84[3];
cx q84[1],q84[2];
cx q84[1],q84[0];
